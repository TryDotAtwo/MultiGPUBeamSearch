import { env } from "cloudflare:workers";
import { beforeEach, describe, expect, test } from "vitest";
import golden from "../../../configs/cayleypy_results_v2_golden.json";
import v1Golden from "../../../configs/cayleypy_results_v1_golden.json";
import { fetchRequest, type WorkerEnv } from "../src/worker.js";
import { consumeValidationMessage, type QueueMessageLike } from "../src/consumer.js";
import { computeIdempotency } from "../src/ids.js";

const URL="https://ingest.example.test";
const context=():ExecutionContext=>({waitUntil:()=>undefined,passThroughOnException:()=>undefined}) as unknown as ExecutionContext;
const bindings=():WorkerEnv=>({INGEST_MODE:"normal",RESULTS_DB:env.RESULTS_DB,RAW_RESULTS:env.RAW_RESULTS,VALIDATE_QUEUE:{send:async()=>undefined} as unknown as Queue});
const post=(path:string,value:unknown)=>fetchRequest(new Request(`${URL}${path}`,{method:"POST",headers:{"content-type":"application/json","CF-Connecting-IP":"203.0.113.77"},body:JSON.stringify(value)}),bindings(),context());

beforeEach(async()=>{await env.RESULTS_DB.exec("DELETE FROM submissions");await env.RESULTS_DB.exec("DELETE FROM ingest_rate_limits");const listed=await env.RAW_RESULTS.list({prefix:"raw/"});if(listed.objects.length)await env.RAW_RESULTS.delete(listed.objects.map(x=>x.key));});

describe("SLURM v2 route isolation",()=>{
 test("POST /v2/results accepts only v2 and returns shared safe status receipts",async()=>{const response=await post("/v2/results",{schema_version:2,results:[golden.cases[0].envelope]});expect(response.status).toBe(202);const body=await response.json() as any;expect(body.receipts).toHaveLength(1);expect(body.receipts[0].status_url).toMatch(/\/v1\/submissions\/[0-9a-f-]+$/);});
 test("v1 and v2 endpoints reject the other schema",async()=>{expect((await post("/v1/results",{schema_version:2,results:[golden.cases[0].envelope]})).status).toBe(400);expect((await post("/v2/results",{schema_version:1,results:[v1Golden.cases[0].envelope]})).status).toBe(400);});
 test("accepts a bounded gzip v2 batch", async () => {
   const bytes = new TextEncoder().encode(JSON.stringify({ schema_version: 2, results: [golden.cases[0].envelope] }));
   const gzip = new CompressionStream("gzip");
   const compressed = new Response(new Blob([bytes]).stream().pipeThrough(gzip)).body;
   const response = await fetchRequest(new Request(`${URL}/v2/results`, {
     method: "POST",
     headers: { "content-type": "application/gzip", "CF-Connecting-IP": "203.0.113.79" },
     body: compressed,
   }), bindings(), context());
   expect(response.status).toBe(202);
 });
 test("v2 rejects fake Kaggle fields instead of requiring them",async()=>{const envelope:any=structuredClone(golden.cases[0].envelope);envelope.kaggle={owner:"fake",slug:"fake",version:1,notebook_sha256:"0".repeat(64)};expect((await post("/v2/results",{schema_version:2,results:[envelope]})).status).toBe(400);});
 test("queue consumer revalidates v2 and enqueues the shared writer",async()=>{let writerId="";const workerEnv:WorkerEnv={...bindings(),GITHUB_WRITER:{getByName:()=>({enqueueValidated:async(id:string)=>{writerId=id;}})}};const accepted=await fetchRequest(new Request(`${URL}/v2/results`,{method:"POST",headers:{"content-type":"application/json","CF-Connecting-IP":"203.0.113.78"},body:JSON.stringify({schema_version:2,results:[golden.cases[0].envelope]})}),workerEnv,context());const receipt=(await accepted.json() as any).receipts[0];const probe:QueueMessageLike&{acked:number;retried:number}={body:{submission_id:receipt.submission_id},attempts:1,acked:0,retried:0,ack(){this.acked++},retry(){this.retried++}};await consumeValidationMessage(probe,workerEnv as any,"normal");expect(probe).toMatchObject({acked:1,retried:0});expect(writerId).toBe(receipt.submission_id);}); test("preserves v2 idempotency and raw/v2 storage",async()=>{const first=await post("/v2/results",{schema_version:2,results:[golden.cases[0].envelope]});const second=await post("/v2/results",{schema_version:2,results:[golden.cases[0].envelope]});const a=(await first.json() as any).receipts[0];const b=(await second.json() as any).receipts[0];expect(a.submission_id).toBe(b.submission_id);const row=await env.RESULTS_DB.prepare("SELECT raw_r2_key FROM submissions WHERE submission_id=?").bind(a.submission_id).first<{raw_r2_key:string}>();expect(row?.raw_r2_key).toMatch(/^raw\/v2\//);});
 test("duplicate v2 queue delivery is idempotent", async () => {
   const writerIds: string[] = [];
   const workerEnv: WorkerEnv = { ...bindings(), GITHUB_WRITER: { getByName: () => ({ enqueueValidated: async (id: string) => { writerIds.push(id); } }) } };
   const accepted = await fetchRequest(new Request(`${URL}/v2/results`, {
     method: "POST",
     headers: { "content-type": "application/json", "CF-Connecting-IP": "203.0.113.83" },
     body: JSON.stringify({ schema_version: 2, results: [golden.cases[0].envelope] }),
   }), workerEnv, context());
   const receipt = (await accepted.json() as any).receipts[0];
   for (const attempts of [1, 2]) {
     const message: QueueMessageLike = { body: { submission_id: receipt.submission_id }, attempts, ack() {}, retry() { throw new Error("unexpected retry"); } };
     await consumeValidationMessage(message, workerEnv as any, "normal");
   }
   expect(writerIds).toEqual([receipt.submission_id, receipt.submission_id]);
 });
 test("v2 obeys store_only, reject, missing, and unknown modes",async()=>{for(const mode of ["reject",undefined,"unknown"]){const e:any={...bindings(),INGEST_MODE:mode};const response=await fetchRequest(new Request(`${URL}/v2/results`,{method:"POST",headers:{"content-type":"application/json","CF-Connecting-IP":`203.0.113.${mode?80:81}`},body:JSON.stringify({schema_version:2,results:[golden.cases[0].envelope]})}),e,context());expect(response.status).toBe(503);}const stored=await fetchRequest(new Request(`${URL}/v2/results`,{method:"POST",headers:{"content-type":"application/json","CF-Connecting-IP":"203.0.113.82"},body:JSON.stringify({schema_version:2,results:[golden.cases[0].envelope]})}),{...bindings(),INGEST_MODE:"store_only"},context());expect(stored.status).toBe(202);});
 test("100 concurrent mixed v1 and v2 publishers retain all receipts",async()=>{const requests=await Promise.all(Array.from({length:100},async(_,index)=>{const isV2=index%2===1;const envelope:any=structuredClone(isV2?golden.cases[0].envelope:v1Golden.cases[0].envelope);envelope.client_submission_id=`018f7a24-8f6b-7c8e-9d1b-${(0x300000000000n+BigInt(index)).toString(16).padStart(12,"0")}`;envelope.run_id=`mixed-${index}`;if(isV2)envelope.provenance.run_id=envelope.run_id;envelope.puzzle_id=1000+index;envelope.idempotency_key=await computeIdempotency(envelope);return fetchRequest(new Request(`${URL}/v${isV2?2:1}/results`,{method:"POST",headers:{"content-type":"application/json","CF-Connecting-IP":`198.51.${Math.floor(index/250)}.${index%250+1}`},body:JSON.stringify({schema_version:isV2?2:1,results:[envelope]})}),bindings(),context());}));expect(requests.map(x=>x.status)).toEqual(Array(100).fill(202));const count=await env.RESULTS_DB.prepare("SELECT COUNT(*) AS count FROM submissions").first<number>("count");expect(count).toBe(100);});});