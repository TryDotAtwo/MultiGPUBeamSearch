import { expect, test } from "vitest";
import { forwardIngest } from "../src/ingest-transport.js";
const request=(body:ReadableStream<Uint8Array>)=>new Request("https://ingest.example.test/v1/results",{method:"POST",body});
test("relay cancels upstream when stub fetch rejects before reading",async()=>{
 let cancelled=false;
 const input=request(new ReadableStream<Uint8Array>({cancel(){cancelled=true;}}));
 await expect(forwardIngest(input,async()=>{throw new Error("stub_failed");})).rejects.toThrow("stub_failed");
 expect(cancelled).toBe(true);
});
test("relay preserves all bytes at EOF without truncation or spurious cancellation",async()=>{
 let cancelled=false;
 const input=request(new ReadableStream<Uint8Array>({start(c){for(const text of ["a","б","c"])c.enqueue(new TextEncoder().encode(text));c.close();},cancel(){cancelled=true;}}));
 const response=await forwardIngest(input,async forwarded=>new Response(await forwarded.text()));
 expect(await response.text()).toBe("aбc");expect(cancelled).toBe(false);
});
test("relay propagates downstream cancellation during pending read",async()=>{
 let cancelled=false;
 const input=request(new ReadableStream<Uint8Array>({cancel(){cancelled=true;}}));
 await forwardIngest(input,async forwarded=>{const reader=forwarded.body!.getReader();const pending=reader.read();await reader.cancel();await pending;return new Response("busy",{status:503});});
 expect(cancelled).toBe(true);
});
