"""LangGraph procurement agents with evidence-first answer construction."""

from __future__ import annotations

from typing import Any, TypedDict

from .core import (
    EvidenceChunk,
    heuristic_risk,
    resolve_entities,
    resolve_conflicting_evidence,
    retrieve_by_embedding,
    validate_citations,
)


class AgentState(TypedDict, total=False):
    project_id: str
    question: str
    package_context: dict[str, Any]
    risk: dict[str, Any]
    evidence: list[dict[str, Any]]
    conflicts: dict[str, Any]
    answer: dict[str, Any]
    errors: list[str]


class ProcurementGraph:
    """Build a graph whose deterministic agents never invent unsupported claims."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        chunks: list[EvidenceChunk],
        embedder: Any,
        *,
        automl_models: dict[str, Any] | None = None,
        model_uris: dict[str, str] | None = None,
    ) -> None:
        self.records = records
        self.chunks = chunks
        self.embedder = embedder
        self.automl_models = automl_models or {}
        self.model_uris = model_uris or {}
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(AgentState)
        workflow.add_node("risk_agent", self._risk_agent)
        workflow.add_node("evidence_agent", self._evidence_agent)
        workflow.add_node("evidence_adjudicator", self._adjudicate)
        workflow.add_node("faithfulness_guard", self._guard)
        workflow.set_entry_point("risk_agent")
        workflow.add_edge("risk_agent", "evidence_agent")
        workflow.add_edge("evidence_agent", "evidence_adjudicator")
        workflow.add_edge("evidence_adjudicator", "faithfulness_guard")
        workflow.add_edge("faithfulness_guard", END)
        return workflow.compile()

    def invoke(
        self,
        project_id: str,
        question: str,
        package_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not project_id.strip() or not question.strip():
            raise ValueError("project_id and question must not be empty.")
        return self.graph.invoke(
            {
                "project_id": project_id.strip(),
                "question": question.strip(),
                "package_context": package_context or {},
                "errors": [],
            }
        )

    def _risk_agent(self, state: AgentState) -> AgentState:
        records = [row for row in self.records if str(row.get("project_id")) == state["project_id"]]
        if records:
            record = records[0]
            risk = heuristic_risk(record)
            source_uri = str(record.get("source_uri") or "unspecified_source")
            record_id = str(record.get("source_record_id") or "unidentified_record")
            citations = [f"[PROCUREMENT | {record_id} | {source_uri}]"]
            claims = [
                {"text": f"Historical rule-based risk classification is {risk['risk_level']}.", "citations": citations},
                {"text": f"Assessment method is {risk['method']}.", "citations": citations},
                {
                    "text": f"Historical heuristic score is {risk['risk_score']:.4f}.",
                    "citations": citations,
                },
            ]
            claims.extend(
                {"text": f"Observed historical signal: {reason}.", "citations": citations}
                for reason in risk["reasons"]
            )
            predictors = {**record, **state.get("package_context", {})}
        else:
            risk = {"status": "ABSTAINED", "reason": "No matching historical procurement record."}
            citations = []
            claims = []
            predictors = dict(state.get("package_context", {}))
        model_predictions = self._model_predictions(predictors)
        model_citations = [
            f"[MODEL | {name} | {self.model_uris.get(name, 'configured MLflow model')}]"
            for name, model in self.automl_models.items()
            if name in model_predictions
        ]
        if model_predictions:
            for name, prediction in model_predictions.items():
                citations_for_model = [
                    citation for citation in model_citations if f"| {name} |" in citation
                ]
                claims.append(
                    {
                        "text": f"Model estimate for {name} is {prediction:.4f}.",
                        "citations": citations_for_model,
                    }
                )
            citations.extend(model_citations)
        return {
            "risk": {
                **risk,
                "status": "GROUNDED" if citations else "ABSTAINED",
                "project_id": state["project_id"],
                "citations": citations,
                "claims": claims,
                "predictions": model_predictions,
                "disclaimer": "Rule-based flags describe observed historical outcomes; ML estimates require a trained and registered model and are decision-support only.",
            }
        }

    def _model_predictions(self, record: dict[str, Any]) -> dict[str, Any]:
        if not self.automl_models:
            return {}
        import pandas as pd

        output: dict[str, Any] = {}
        for key, model in self.automl_models.items():
            predictors = {
                name: value
                for name, value in record.items()
                if name not in {
                    "actual_days",
                    "amendment_count",
                    "outcome",
                    "source_record_id",
                    "source_uri",
                }
            }
            features = pd.DataFrame([predictors])
            if key == "timeline_days":
                output[key] = max(0.0, float(model.predict(features)[0]))
            elif key in {"amendment_probability", "rebid_probability"}:
                output[key] = float(model.predict(features)[0])
        return output

    def _evidence_agent(self, state: AgentState) -> AgentState:
        vector = self.embedder.encode(state["question"])
        historical_rows = [
            row for row in self.records if str(row.get("project_id")) == state["project_id"]
        ]
        project = (
            historical_rows[0]
            if historical_rows
            else {"project_id": state["project_id"], **state.get("package_context", {})}
        )
        document_metadata = {
            chunk.document_id: {
                "document_id": chunk.document_id,
                "project_id": chunk.project_id,
                "sector_code": chunk.sector_code,
                "document_year": chunk.document_year,
            }
            for chunk in self.chunks
        }
        entity_matches = resolve_entities([project], list(document_metadata.values()))
        related_documents = {match.document_id for match in entity_matches}
        entity_match_details = {
            match.document_id: {
                "method": match.method,
                "confidence": match.confidence,
                "year_distance": match.year_distance,
            }
            for match in entity_matches
        }
        matches = retrieve_by_embedding(
            vector,
            self.chunks,
            document_ids=related_documents,
            minimum_score=0.15,
            limit=5,
        )
        evidence = [
            {
                "chunk_text": chunk.chunk_text,
                "citation_id": chunk.citation_id,
                "document_id": chunk.document_id,
                "document_type": chunk.document_type,
                "project_id": chunk.project_id,
                "effective_date": chunk.effective_date,
                "score": score,
                "claim_key": chunk.claim_key,
                "claim_value": chunk.claim_value,
                "entity_resolution": entity_match_details.get(chunk.document_id),
            }
            for chunk, score in matches
        ]
        return {"evidence": evidence}

    def _adjudicate(self, state: AgentState) -> AgentState:
        return {"conflicts": resolve_conflicting_evidence(state.get("evidence", []))}

    def _guard(self, state: AgentState) -> AgentState:
        risk = state.get("risk", {"status": "ABSTAINED"})
        evidence = state.get("evidence", [])
        allowed = {item["citation_id"] for item in evidence}
        allowed.update(risk.get("citations", []))
        claims = list(risk.get("claims", []))
        errors = validate_citations(claims, allowed)
        if errors:
            return {
                "errors": errors,
                "answer": {
                    "status": "BLOCKED",
                    "answer": "Response blocked because one or more factual claims did not pass source validation.",
                    "claims": [],
                    "citations": [],
                    "human_review_required": True,
                },
            }

        citations = list(dict.fromkeys(risk.get("citations", []) + [item["citation_id"] for item in evidence]))
        if risk.get("status") != "GROUNDED" and not evidence:
            return {
                "answer": {
                    "status": "ABSTAINED",
                    "answer": "No verified procurement record or sufficiently relevant source document was found.",
                    "claims": [],
                    "citations": [],
                    "human_review_required": True,
                }
            }
        answer_parts = []
        if risk.get("status") == "GROUNDED":
            if "risk_level" in risk:
                answer_parts.append(
                    f"Historical rule-based risk flag: {risk['risk_level']} "
                    f"({risk['method']}). This is a decision-support heuristic, not a trained probability."
                )
            else:
                answer_parts.append(
                    "Model estimates are decision-support outputs, not procurement decisions."
                )
            for name, value in risk.get("predictions", {}).items():
                label = "Predicted completion days" if name == "timeline_days" else f"Estimated {name.replace('_', ' ')}"
                answer_parts.append(f"{label}: {value:.3f}.")
        if evidence:
            answer_parts.append(
                "Retrieved historical evidence (review the cited passage): "
                + " ".join(
                    f"{item['chunk_text']} {item['citation_id']}"
                    for item in evidence
                )
            )
        conflicts = state.get("conflicts", {})
        if conflicts.get("status") == "CONFLICTING_EVIDENCE":
            answer_parts.append(
                "Conflicting sources describe the same claim differently; both are retained and human review is required."
            )
        return {
            "answer": {
                "status": "GROUNDED",
                "answer": "\n\n".join(answer_parts),
                "claims": claims,
                "citations": citations,
                "risk": risk,
                "evidence": evidence,
                "conflicts": conflicts,
                "human_review_required": conflicts.get("status") == "CONFLICTING_EVIDENCE",
            }
        }
