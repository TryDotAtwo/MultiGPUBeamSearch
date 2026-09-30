import { env } from "cloudflare:workers";
import { beforeEach, expect, test } from "vitest";
import golden from "../../../configs/cayleypy_results_v1_golden.json";
import { fetchRequest, INGRESS_OBJECT_NAME, type WorkerEnv } from "../src/worker.js";
const ctx = {} as ExecutionContext;
const bindings = env as unknown as WorkerEnv;
const request = (body: BodyInit, type = "application/json") => new Request("https://ingest.example.test/v1/results", { method: "POST", headers: { "content-type": type, "CF-Connecting-IP": "203.0.113.81" }, body });
beforeEach(async () => { await env.RESULTS_DB.exec("DELETE FROM submissions"); await env.RESULTS_DB.exec("DELETE FROM ingest_rate_limits"); });
test("real DO fetch preserves synchronous receipt and duplicate identity", async () => {
 const body = JSON.stringify({ schema_version: 1, results: [golden.cases[0].envelope] });
 const a = await fetchRequest(request(body), bindings, ctx);
 expect(a.status).toBe(202);
 const first = await a.json() as any;
 const b = await fetchRequest(request(body), bindings, ctx);
 expect(b.status).toBe(202);
 expect((await b.json() as any).receipts).toEqual(first.receipts);
 expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS n FROM submissions").first("n")).toBe(1);
});
test("real DO validates entire archive before storing any member", async () => {
 const bad = structuredClone(golden.cases[0].envelope); bad.idempotency_key = "0".repeat(64);
 const bytes = new TextEncoder().encode(JSON.stringify({ schema_version: 1, results: [golden.cases[0].envelope, bad] }));
 const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
 const response = await fetchRequest(request(stream, "application/gzip"), bindings, ctx);
 expect(response.status).toBe(400);
 expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS n FROM submissions").first("n")).toBe(0);
});
test("edge only forwards bounded native stream and targets fixed ingress object", async () => {
 let name = ""; let forwarded: Request | undefined;
 const original = request("{unparsed}");
 const response = new Response("unchanged", { status: 202 });
 const testEnv: WorkerEnv = { ...bindings, GITHUB_WRITER: { getByName: (value) => { name = value; return { enqueueValidated: async () => undefined, fetch: async (input) => { forwarded = input; expect(input.bodyUsed).toBe(false); return response; } }; } } };
 expect(await fetchRequest(original, testEnv, ctx)).toBe(response);
 expect(name).toBe(INGRESS_OBJECT_NAME); expect(forwarded!.url).toBe(original.url);
});
test("missing DO fails closed", async () => {
 const response = await fetchRequest(request("{}"), { ...bindings, GITHUB_WRITER: undefined }, ctx);
 expect(response.status).toBe(503); expect(await response.json()).toEqual({ error: "ingest_unavailable" });
});

test("real DO bounds gzip expansion and rejects record overflow without acceptance", async () => {
 for (const body of [" ".repeat(4 * 1024 * 1024 + 1), JSON.stringify({schema_version:1, results:Array(101).fill(golden.cases[0].envelope)})]) {
  const compressed = new Blob([body]).stream().pipeThrough(new CompressionStream("gzip"));
  const response = await fetchRequest(request(compressed, "application/gzip"), bindings, ctx);
  expect(response.status).toBe(413);
 }
 expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS n FROM submissions").first("n")).toBe(0);
});
test("real DO rejects interleaved ingress with bounded retryable busy response", async () => {
 let release!: () => void;
 let reading!: () => void;
 const began = new Promise<void>((resolve) => { reading = resolve; });
 const held = new Promise<void>((resolve) => { release = resolve; });
 const stream = new ReadableStream<Uint8Array>({ async pull(controller) { reading(); await held; controller.enqueue(new TextEncoder().encode("{}")); controller.close(); } });
 const first = fetchRequest(request(stream), bindings, ctx);
 await began;
 // IP admission is asynchronous; force a real DO-side admission signal by issuing
 // a direct same-object request while the first stream is held.
 const stub = bindings.GITHUB_WRITER!.getByName(INGRESS_OBJECT_NAME);
 let busy: Response | undefined;
 for (let attempt=0; attempt<10; attempt++) {
  busy = await stub.fetch!(request("{}"));
  if (busy.status === 503) break;
 }
 expect(busy!.status).toBe(503); expect(await busy!.json()).toEqual({error:"ingest_busy"});
 release(); expect((await first).status).toBe(400);
});

test("real DO releases stalled body admission on deadline, then accepts next valid request", async () => {
 let cancelled = false;
 const stalled = new ReadableStream<Uint8Array>({ cancel() { cancelled = true; } });
 const response = await fetchRequest(request(stalled), bindings, ctx);
 expect(response.status).toBe(408); expect(await response.json()).toEqual({error:"body_read_timeout"});
 await new Promise(resolve => setTimeout(resolve, 10));
 expect(cancelled).toBe(true);
 expect((await fetchRequest(request(JSON.stringify({schema_version:1,results:[golden.cases[0].envelope]})),bindings,ctx)).status).toBe(202);
}, 20_000);
test("real DO bounds malformed batch diagnostics before exhaustive schema work", async () => {
 const response = await fetchRequest(request(JSON.stringify({schema_version:1,results:Array(500_000).fill({})})),bindings,ctx);
 expect(response.status).toBe(413);
 const invalid = structuredClone(golden.cases[0].envelope) as any;
 invalid.proof.initial_state = Array(500_000).fill("x");
 const nested = await fetchRequest(request(JSON.stringify({schema_version:1,results:[invalid]})),bindings,ctx);
 expect(nested.status).toBe(400);
 expect((await nested.json() as any).errors).toHaveLength(1);
 expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS n FROM submissions").first("n")).toBe(0);
});

test("real DO enforces actual compressed cap with missing and dishonest lengths and cancels upstream", async () => {
 for (const declared of [null, "1"]) {
  let sent = 0; let cancelled = false;
  const chunk = new Uint8Array(64 * 1024).fill(65);
  const stream = new ReadableStream<Uint8Array>({
   start(controller) { controller.enqueue(new Uint8Array([31,139,8,8,0,0,0,0,0,3])); },
   pull(controller) { sent++; controller.enqueue(chunk); if (sent>520) controller.close(); },
   cancel() { cancelled=true; },
  });
  const input = request(stream, "application/gzip"); if (declared!==null) input.headers.set("content-length",declared);
  const response = await fetchRequest(input,bindings,ctx);
  expect({status:response.status,sent,body:await response.clone().json()}).toEqual({status:413,sent:expect.any(Number),body:{error:"request_too_large"}});
  await new Promise(resolve=>setTimeout(resolve,10)); expect(cancelled).toBe(true);
  expect((await fetchRequest(request("{}"),bindings,ctx)).status).toBe(400);
 }
 expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS n FROM submissions").first("n")).toBe(0);
});

test("real DO deadline is total across slow multiple chunks", async () => {
 let timer: ReturnType<typeof setTimeout>; let pulls=0; let cancelled=false;
 const stream = new ReadableStream<Uint8Array>({
  pull(controller) { return new Promise<void>((resolve) => { timer=setTimeout(()=>{ if (!cancelled) controller.enqueue(new TextEncoder().encode(" ")); resolve(); }, ++pulls === 1 ? 10_000 : 10_000); }); },
  cancel() { cancelled=true; clearTimeout(timer); },
 });
 const began=Date.now();
 const response=await fetchRequest(request(stream),bindings,ctx);
 expect(response.status).toBe(408); expect(Date.now()-began).toBeLessThan(18_000);
 expect(cancelled).toBe(true);
 expect((await fetchRequest(request("{}"),bindings,ctx)).status).toBe(400);
},20_000);
