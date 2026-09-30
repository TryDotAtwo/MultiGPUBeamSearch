import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import v1 from "../../../configs/cayleypy_results_schema_v1.json";
import v2 from "../../../configs/cayleypy_results_schema_v2.json";
import type { ResultEnvelope, SchemaVersion } from "./schema-dispatch.js";
const ajv = new Ajv2020({ allErrors: false, strict: true, strictRequired: false, strictTypes: false, allowUnionTypes: true });
addFormats(ajv);
const validators = { 1: ajv.compile(v1), 2: ajv.compile(v2) };
/** Same complete schemas, bounded first-error diagnostics for public ingress. */
export function validateIngressBatch(value: unknown, version: SchemaVersion): {ok:true;value:{results:ResultEnvelope[]}} | {ok:false;errors:{path:string;keyword:string}[]} {
 const validate = validators[version];
 if (!validate(value)) return {ok:false,errors:(validate.errors ?? []).slice(0,1).map(error=>({path:error.instancePath.slice(0,256),keyword:error.keyword}))};
 return {ok:true,value:value as {results:ResultEnvelope[]}};
}
