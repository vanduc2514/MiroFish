"""Ontology generation service."""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.ontology_normalizer import normalize_ontology_for_zep



ONTOLOGY_SYSTEM_PROMPT = """You are an expert knowledge-graph ontology designer. Your task is to analyze the provided text and simulation requirement, then design entity types and relationship types suitable for a **social media opinion simulation**.

**Important: you must output valid JSON only. Do not output any additional text.**

## Core task background

We are building a **social media opinion simulation system**. In this system:
- Each entity is an account or actor that can speak, interact, and spread information on social media.
- Entities influence one another, repost, comment, and respond.
- We need to simulate how different parties react during a public-opinion event and how information spreads.

Therefore, **entities must be real-world actors that can speak and interact on social media**:

**Allowed**:
- Specific individuals such as public figures, people directly involved, opinion leaders, scholars, experts, or ordinary people
- Companies and businesses, including their official accounts
- Organizations and institutions, such as universities, associations, NGOs, and unions
- Government departments and regulators
- Media organizations, such as newspapers, TV stations, self-media accounts, and websites
- Social media platforms themselves
- Representatives of specific groups, such as alumni associations, fan groups, or rights-advocacy groups

**Not allowed**:
- Abstract concepts such as "public opinion", "emotion", or "trend"
- Topics or themes such as "academic integrity" or "education reform"
- Positions or attitudes such as "supporters" or "opponents"

## Output format

Output JSON in the following structure:

```json
{
    "entity_types": [
        {
            "name": "Entity type name (English, PascalCase)",
            "description": "Short description (English, under 100 characters)",
            "attributes": [
                {
                    "name": "Attribute name (English, snake_case)",
                    "type": "text",
                    "description": "Attribute description"
                }
            ],
            "examples": ["Example entity 1", "Example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Relationship type name (English, UPPER_SNAKE_CASE)",
            "description": "Short description (English, under 100 characters)",
            "source_targets": [
                {"source": "Source entity type (must exactly match an entity type name)", "target": "Target entity type (must exactly match an entity type name)"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief analysis of the content (English)"
}
```

## Design guidelines (very important)

### 1. Entity type design - must be followed strictly

**Quantity requirement: exactly 10 entity types**

**Hierarchy requirement (must include both specific and fallback types)**:

Your 10 entity types must follow this structure:

A. **Fallback types (must be included and placed as the last 2 items)**:
   - `Person`: Fallback type for any individual human being. Use this when a person does not belong to a more specific person category.
   - `Organization`: Fallback type for any organization or institution. Use this when an organization does not belong to a more specific organization category.

B. **Specific types (8, designed based on the text)**:
   - Design more specific types for the main roles that appear in the text.
   - Example: for an academic event, types might include `Student`, `Professor`, and `University`.
   - Example: for a business event, types might include `Company`, `CEO`, and `Employee`.

**Why fallback types are needed**:
- The text may mention many kinds of people, such as school teachers, bystanders, or anonymous netizens.
- If no specific type fits them, they should be classified as `Person`.
- Likewise, small organizations or temporary groups should fall under `Organization`.

**Specific type design principles**:
- Identify high-frequency or important role categories from the text.
- Each specific type should have a clear boundary and avoid overlap.
- The description must clearly explain how this type differs from the fallback type.

### 2. Relationship type design

- Quantity: 6-10 relationship types
- Relationships should reflect realistic social-media interactions and ties
- Make sure the `source_targets` cover the entity types you define

### 3. Attribute design

- Each entity type should have 1-3 key attributes
- **Important**: attribute names cannot use `name`, `uuid`, `group_id`, `created_at`, or `summary` because these are reserved system fields
- Recommended names include `full_name`, `title`, `role`, `position`, `location`, and `description`

## Entity type references

**Person types (specific)**:
- Student: students
- Professor: professors or scholars
- Journalist: journalists
- Celebrity: celebrities or influencers
- Executive: business executives
- Official: government officials
- Lawyer: lawyers
- Doctor: doctors

**Person type (fallback)**:
- Person: any human individual not covered by a more specific person type

**Organization types (specific)**:
- University: universities
- Company: companies and businesses
- GovernmentAgency: government agencies
- MediaOutlet: media organizations
- Hospital: hospitals
- School: primary and secondary schools
- NGO: non-governmental organizations

**Organization type (fallback)**:
- Organization: any organization not covered by a more specific organization type

## Relationship type references

- WORKS_FOR: works for
- STUDIES_AT: studies at
- AFFILIATED_WITH: is affiliated with
- REPRESENTS: represents
- REGULATES: regulates
- REPORTS_ON: reports on
- COMMENTS_ON: comments on
- RESPONDS_TO: responds to
- SUPPORTS: supports
- OPPOSES: opposes
- COLLABORATES_WITH: collaborates with
- COMPETES_WITH: competes with
"""


class OntologyGenerator:
    """Ontology Generator."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate the requested object."""
        
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        
        result = self._validate_and_process(result)
        
        return result
    
    
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Build user message."""
        
        
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += (
                f"\n\n...(Original text length: {original_length} characters. "
                f"Only the first {self.MAX_TEXT_LENGTH_FOR_LLM} characters were used for ontology analysis.)..."
            )
        
        message = f"""## Simulation Requirement

{simulation_requirement}

## Document Content

{combined_text}
"""
        
        if additional_context:
            message += f"""
## Additional Notes

{additional_context}
"""
        
        message += """
Based on the content above, design entity types and relationship types suitable for a social-opinion simulation.

**Rules you must follow**:
1. You must output exactly 10 entity types.
2. The last 2 entity types must be the fallback types: `Person` and `Organization`.
3. The first 8 entity types must be specific types designed from the text.
4. All entity types must be real-world actors that can speak or interact, not abstract concepts.
5. Attribute names cannot use reserved fields such as `name`, `uuid`, or `group_id`; use alternatives like `full_name` or `org_name`.
6. Entity type names must contain only letters and numbers. They cannot contain underscores, spaces, or hyphens. For example, `StudentLeader` is valid but `Student_Leader` is not.
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate And Process."""
        
        
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10
        
        
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            
            result["entity_types"].extend(fallbacks_to_add)
        
        
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        normalized_result, _ = normalize_ontology_for_zep(result)
        return normalized_result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """Generate python code."""
        code_lines = [
            '"""',
            'Custom entity type definitions',
            'Auto-generated by MiroFish for social-opinion simulation',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Entity Type Definitions ==============',
            '',
        ]
        
        
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== Relationship Type Definitions ==============')
        code_lines.append('')
        
        
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        
        code_lines.append('# ============== Type Configuration ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)
