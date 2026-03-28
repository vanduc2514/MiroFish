"""Intelligent simulation configuration generator."""

import json
import math
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.simulation_config')


CHINA_TIMEZONE_CONFIG = {
    
    "dead_hours": [0, 1, 2, 3, 4, 5],
    
    "morning_hours": [6, 7, 8],
    
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    
    "peak_hours": [19, 20, 21, 22],
    
    "night_hours": [23],
    
    "activity_multipliers": {
        "dead": 0.05,      
        "morning": 0.4,    
        "work": 0.7,       
        "peak": 1.5,       
        "night": 0.5       
    }
}


@dataclass
class AgentActivityConfig:
    """Agent Activity Config."""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str
    
    
    activity_level: float = 0.5  
    
    
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    
    
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    
    
    response_delay_min: int = 5
    response_delay_max: int = 60
    
    
    sentiment_bias: float = 0.0
    
    
    stance: str = "neutral"  # supportive, opposing, neutral, observer
    
    
    influence_weight: float = 1.0


@dataclass  
class TimeSimulationConfig:
    """Time Simulation Config."""
    
    total_simulation_hours: int = 72  
    
    
    minutes_per_round: int = 60
    
    
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20
    
    
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5
    
    
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  
    
    
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4
    
    
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Event Config."""
    
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    
    
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    
    
    hot_topics: List[str] = field(default_factory=list)
    
    
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Platform Config."""
    platform: str  # twitter or reddit
    
    
    recency_weight: float = 0.4  
    popularity_weight: float = 0.3  
    relevance_weight: float = 0.3  
    
    
    viral_threshold: int = 10
    
    
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Simulation Parameters."""
    
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str
    
    
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    
    
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    
    
    event_config: EventConfig = field(default_factory=EventConfig)
    
    
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None
    
    
    llm_model: str = ""
    llm_base_url: str = ""
    
    
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary."""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert the object to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """Simulation Config Generator."""
    
    
    MAX_CONTEXT_LENGTH = 50000
    
    AGENTS_PER_BATCH = 15
    
    
    TIME_CONFIG_CONTEXT_LENGTH = 10000   
    EVENT_CONFIG_CONTEXT_LENGTH = 8000   
    ENTITY_SUMMARY_LENGTH = 300          
    AGENT_SUMMARY_LENGTH = 300           
    ENTITIES_PER_TYPE_DISPLAY = 20       
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """Generate config."""
        logger.info(f"开始智能生成模拟配置: simulation_id={simulation_id}, 实体数={len(entities)}")
        
        
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  
        current_step = 0
        
        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")
        
        
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        reasoning_parts = []
        
        
        report_progress(1, "生成时间配置...")
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"时间配置: {time_config_result.get('reasoning', '成功')}")
        
        
        report_progress(2, "生成事件配置和热点话题...")
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"事件配置: {event_config_result.get('reasoning', '成功')}")
        
        
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]
            
            report_progress(
                3 + batch_idx,
                f"生成Agent配置 ({start_idx + 1}-{end_idx}/{len(entities)})..."
            )
            
            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)
        
        reasoning_parts.append(f"Agent配置: 成功生成 {len(all_agent_configs)} 个")
        
        
        logger.info("为初始帖子分配合适的发布者 Agent...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(f"初始帖子分配: {assigned_count} 个帖子已分配发布者")
        
        
        report_progress(total_steps, "生成平台配置...")
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info(f"模拟配置生成完成: {len(params.agent_configs)} 个Agent配置")
        
        return params
    
    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """Build context."""
        
        
        entity_summary = self._summarize_entities(entities)
        
        
        context_parts = [
            f"## Simulation requirement\n{simulation_requirement}",
            f"\n## Entity information ({len(entities)})\n{entity_summary}",
        ]
        
        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500  
        
        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(document truncated)"
            context_parts.append(f"\n## Source document content\n{doc_text}")
        
        return "\n".join(context_parts)
    
    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Summarize Entities."""
        lines = []
        
        
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)})")
            
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... {len(type_entities) - display_count} more")
        
        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Call llm with retry."""
        import re
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  
                    
                )
                
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                
                if finish_reason == 'length':
                    logger.warning(f"LLM输出被截断 (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)
                
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败 (attempt {attempt+1}): {str(e)[:80]}")
                    
                    
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    
                    last_error = e
                    
            except Exception as e:
                logger.warning(f"LLM调用失败 (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))
        
        raise last_error or Exception("LLM调用失败")
    
    def _fix_truncated_json(self, content: str) -> str:
        """Fix truncated json."""
        content = content.strip()
        
        
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        
        if content and content[-1] not in '",}]':
            content += '"'
        
        
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Try fix config json."""
        import re
        
        
        content = self._fix_truncated_json(content)
        
        
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s
            
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)
            
            try:
                return json.loads(json_str)
            except:
                
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass
        
        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Generate time config."""
        
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]
        
        
        max_agents_allowed = max(1, int(num_entities * 0.9))
        
        prompt = f"""Generate a time-simulation configuration for the following scenario.

{context_truncated}

## Task
Return time-configuration JSON.

### Baseline heuristics
- Assume participant behavior broadly follows a China Standard Time daily rhythm unless the scenario strongly suggests otherwise
- Activity is minimal from 00:00-05:00 (activity multiplier about 0.05)
- Activity increases from 06:00-08:00 (about 0.4)
- Work hours from 09:00-18:00 are moderately active (about 0.7)
- Evening from 19:00-22:00 is typically the peak (about 1.5)
- Activity declines after 23:00 (about 0.5)
- These are heuristics only; adapt the exact schedule to the event and participant mix
  - Example: students may peak around 21:00-23:00, media may be active all day, official institutions may stay within work hours
  - Example: breaking-news events may still attract late-night discussion, so off_peak_hours can be shorter

### Return JSON only, no Markdown

Example:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Brief explanation for this schedule"
}}

Field notes:
- total_simulation_hours (int): total duration, usually 24-168 hours
- minutes_per_round (int): duration per round, usually 30-120 minutes, commonly 60
- agents_per_hour_min (int): minimum active agents per hour, range 1-{max_agents_allowed}
- agents_per_hour_max (int): maximum active agents per hour, range 1-{max_agents_allowed}
- peak_hours (int array): peak activity windows
- off_peak_hours (int array): low-activity windows, usually late night and early morning
- morning_hours (int array): early-day activity window
- work_hours (int array): workday activity window
- reasoning (string): brief explanation for the configuration"""

        system_prompt = (
            "You are an expert in social-media simulation design. "
            "Return pure JSON only. All free-text fields must be written in English."
        )
        
        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"时间配置LLM生成失败: {e}, 使用默认配置")
            return self._get_default_time_config(num_entities)
    
    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Get default time config."""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Using the default China Standard Time activity rhythm with 1-hour rounds"
        }
    
    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """Parse time config."""
        
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))
        
        
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) 超过总Agent数 ({num_entities})，已修正")
            agents_per_hour_min = max(1, num_entities // 10)
        
        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) 超过总Agent数 ({num_entities})，已修正")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)
        
        
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max，已修正为 {agents_per_hour_min}")
        
        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),  
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )

    @staticmethod
    def _contains_cjk(value: Any) -> bool:
        if isinstance(value, str):
            return bool(re.search(r'[\u4e00-\u9fff]', value))
        if isinstance(value, list):
            return any(SimulationConfigGenerator._contains_cjk(item) for item in value)
        if isinstance(value, dict):
            return any(SimulationConfigGenerator._contains_cjk(item) for item in value.values())
        return False

    def _ensure_event_config_english(self, result: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "hot_topics": result.get("hot_topics", []),
            "narrative_direction": result.get("narrative_direction", ""),
            "initial_posts": [
                {
                    "content": post.get("content", ""),
                    "poster_type": post.get("poster_type", ""),
                }
                for post in result.get("initial_posts", [])
            ],
            "reasoning": result.get("reasoning", ""),
        }

        text_fields = [
            payload["hot_topics"],
            payload["narrative_direction"],
            [post.get("content", "") for post in payload["initial_posts"]],
            payload["reasoning"],
        ]
        if not any(self._contains_cjk(field) for field in text_fields):
            return result

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Translate the provided event-configuration text into natural English. "
                            "Return valid JSON only. Preserve the JSON shape. "
                            "Do not change any poster_type values. "
                            "Translate only hot_topics, narrative_direction, initial_posts[].content, and reasoning."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            translated = json.loads(response.choices[0].message.content)
            result["hot_topics"] = translated.get("hot_topics", result.get("hot_topics", []))
            result["narrative_direction"] = translated.get("narrative_direction", result.get("narrative_direction", ""))
            result["reasoning"] = translated.get("reasoning", result.get("reasoning", ""))

            translated_posts = translated.get("initial_posts", [])
            if isinstance(translated_posts, list):
                merged_posts = []
                original_posts = result.get("initial_posts", [])
                for idx, original_post in enumerate(original_posts):
                    translated_post = translated_posts[idx] if idx < len(translated_posts) and isinstance(translated_posts[idx], dict) else {}
                    merged_posts.append({
                        **original_post,
                        "content": translated_post.get("content", original_post.get("content", "")),
                        "poster_type": original_post.get("poster_type", translated_post.get("poster_type", "")),
                    })
                result["initial_posts"] = merged_posts
        except Exception as e:
            logger.warning(f"Failed to normalize event config text to English: {e}")

        return result
    
    def _generate_event_config(
        self, 
        context: str, 
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Generate event config."""
        
        
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))
        
        
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)
        
        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}" 
            for t, examples in type_examples.items()
        ])
        
        
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]
        
        prompt = f"""Generate an event configuration for the following simulation.

Simulation requirement: {simulation_requirement}

{context_truncated}

## Available entity types and examples
{type_info}

## Task
Return event configuration JSON that:
- extracts hot-topic keywords
- describes the likely narrative direction
- drafts initial post content, where **every post must include a poster_type**

Important:
- poster_type must be selected from the available entity types above so each initial post can be assigned to an appropriate agent
- all free-text output must be in English
- poster_type values must remain exact entity-type names

Return JSON only, no Markdown:
{{
    "hot_topics": ["keyword1", "keyword2", "..."],
    "narrative_direction": "<English narrative direction description>",
    "initial_posts": [
        {{"content": "<English post content>", "poster_type": "<entity type from the available list>"}},
        ...
    ],
    "reasoning": "<brief English explanation>"
}}"""

        system_prompt = (
            "You are an expert in social-media narrative analysis. "
            "Return pure JSON only. All free-text fields must be in English. "
            "poster_type values must exactly match one of the available entity types."
        )
        
        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            return self._ensure_event_config_english(result)
        except Exception as e:
            logger.warning(f"事件配置LLM生成失败: {e}, 使用默认配置")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Using default configuration"
            }
    
    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Parse event config."""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """Assign Initial Post Agents."""
        if not event_config.initial_posts:
            return event_config
        
        
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)
        
        
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }
        
        
        used_indices: Dict[str, int] = {}
        
        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")
            
            
            matched_agent_id = None
            
            
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break
            
            
            if matched_agent_id is None:
                logger.warning(f"未找到类型 '{poster_type}' 的匹配 Agent，使用影响力最高的 Agent")
                if agent_configs:
                    
                    sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0
            
            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id
            })
            
            logger.info(f"初始帖子分配: poster_type='{poster_type}' -> agent_id={matched_agent_id}")
        
        event_config.initial_posts = updated_posts
        return event_config
    
    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """Generate agent configs batch."""
        
        
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })
        
        prompt = f"""Generate social-media activity settings for each entity below.

Simulation requirement: {simulation_requirement}

## Entity list
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Task
Generate JSON activity settings for every entity.

Guidance:
- Assume activity generally follows a China Standard Time rhythm unless the scenario suggests otherwise: minimal activity from 00:00-05:00 and peak activity around 19:00-22:00
- Official institutions (University/GovernmentAgency): lower activity (0.1-0.3), mostly active during work hours (09:00-17:00), slower response (60-240 min), higher influence (2.5-3.0)
- Media outlets (MediaOutlet): medium activity (0.4-0.6), active most of the day (08:00-23:00), fast response (5-30 min), higher influence (2.0-2.5)
- Individuals (Student/Person/Alumni): higher activity (0.6-0.9), mostly active in the evening (18:00-23:00), fast response (1-15 min), lower influence (0.8-1.2)
- Public figures and experts: medium activity (0.4-0.6), medium-to-high influence (1.5-2.0)

Return JSON only, no Markdown:
{{
    "agent_configs": [
        {{
            "agent_id": <must exactly match the input>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <posting frequency>,
            "comments_per_hour": <comment frequency>,
            "active_hours": [<active hour list>],
            "response_delay_min": <minimum response delay in minutes>,
            "response_delay_max": <maximum response delay in minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <influence weight>
        }},
        ...
    ]
}}"""

        system_prompt = (
            "You are an expert in social-media behavior modeling. "
            "Return pure JSON only. All free-text fields must be written in English."
        )
        
        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Agent配置批次LLM生成失败: {e}, 使用规则生成")
            llm_configs = {}
        
        
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})
            
            
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)
            
            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)
        
        return configs
    
    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """Generate agent config by rule."""
        entity_type = (entity.get_entity_type() or "Unknown").lower()
        
        if entity_type in ["university", "governmentagency", "ngo"]:
            
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
    
