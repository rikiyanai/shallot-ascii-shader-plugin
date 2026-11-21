/** Cancel every deferred feed BEFORE awaiting an invalid anchor attempt.
 * Otherwise playback waits for tone/labels supplied only by publication. */
export async function discardPartAnchorAttempt<T extends {readonly resident?:{release():void}}>(attempts:readonly {
 readonly played?:Promise<T>;
 readonly cancel:()=>void;
}[]):Promise<readonly PromiseSettledResult<Awaited<T>|undefined>[]>{
 for(const attempt of attempts)attempt.cancel();
 const settled=await Promise.allSettled(attempts.map(attempt=>attempt.played));
 for(const result of settled)if(result.status==='fulfilled')result.value?.resident?.release();
 // A discarded tape may also report another exhausted producer. Its demand
 // must survive even though its drawing is not eligible for publication.
 return settled;
}
