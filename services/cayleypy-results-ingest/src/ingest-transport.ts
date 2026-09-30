/** Zero-copy, demand-driven relay; no decoding, accumulation, cloning or tee.
 * Retain the upstream reader so cancellation is not lost at DO fetch boundary.
 */
export async function forwardIngest(request: Request, fetch: (request: Request) => Promise<Response>): Promise<Response> {
 const reader = request.body?.getReader();
 let finished = reader === undefined;
 const cancel = () => { if (reader && !finished) { finished=true; void reader.cancel().catch(()=>undefined); } };
 const body = reader ? new ReadableStream<Uint8Array>({
  async pull(controller) {
   try {
    const chunk=await reader.read();
    if (chunk.done) { finished=true; controller.close(); }
    else controller.enqueue(chunk.value);
   } catch (error) { controller.error(error); }
  },
  cancel() { cancel(); },
 }, { highWaterMark: 0 }) : null;
 try {
  return await fetch(new Request(request.url, { method:request.method, headers:request.headers, body }));
 } finally {
  cancel();
  try { reader?.releaseLock(); } catch { /* Pending read is cancelled above. */ }
 }
}
