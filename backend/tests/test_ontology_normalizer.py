from app.utils.ontology_normalizer import normalize_ontology_for_zep


def test_normalize_ontology_entity_names_and_source_targets():
    ontology = {
        "entity_types": [
            {
                "name": "IH_Team",
                "description": "Escalation team",
                "attributes": [],
            },
            {
                "name": "billing department",
                "description": "Billing org",
                "attributes": [],
            },
        ],
        "edge_types": [
            {
                "name": "LEADS",
                "description": "Leadership relation",
                "source_targets": [
                    {"source": "IH_Team", "target": "billing department"},
                ],
                "attributes": [],
            }
        ],
    }

    normalized, entity_name_mapping = normalize_ontology_for_zep(ontology)

    assert entity_name_mapping["IH_Team"] == "IHTeam"
    assert entity_name_mapping["billing department"] == "BillingDepartment"
    assert normalized["entity_types"][0]["name"] == "IHTeam"
    assert normalized["entity_types"][1]["name"] == "BillingDepartment"
    assert normalized["edge_types"][0]["source_targets"] == [
        {"source": "IHTeam", "target": "BillingDepartment"}
    ]
