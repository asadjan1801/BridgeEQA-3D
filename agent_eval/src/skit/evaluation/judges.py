"""
LLM and VLM judges for evaluating answer correctness.

Provides text-based and vision-language model judges to evaluate agent answers
against ground truth with caching support.
"""

import asyncio
import hashlib
from typing import List

from agents import Agent, Runner

from .caches import TextCorrectnessCache, VlmCorrectnessCache


# =============================================================================
# Evaluation Prompts
# =============================================================================

CORRECTNESS_PROMPT = """
    You will be given a student answer and a ground truth.
    Your task is to evaluate the student answer by comparing it with the ground truth.
    Give your evaluation on a scale of 0.0 to 1.0, where 0.0 means that the answer is completely unrelated to the ground truth, and 1.0 means that the answer is completely accurate and aligns perfectly with the ground truth.
    In your evaluation do not consider references to filenames or Node numbers in the student's answer or the references to parts or contents of the scene from the teacher's answer in this evaluation as the reference names have been changed between them. Node names and filenames may still correspond to the same object in the scene, as such your evaluation should focus on reasoning about used for the answer and its correctness with respect to the ground truth.

    For instance,
    correctness_score: 0.0: The answer is completely unrelated to the ground truth.
    correctness_score: 0.3: The answer has minor relevance but does not align with the ground truth.
    correctness_score: 0.5: The answer has moderate relevance but contains inaccuracies.
    correctness_score: 0.7: The answer aligns with the reference but has minor errors or omissions.
    correctness_score: 1.0: The answer is completely accurate and aligns perfectly with the ground truth.
    Format the output as a single number as correctness_score: 0.5 for example. Do not produce any other output. output your rating, as a number between 0.0 and 1.0"""

VLM_CORRECTNESS_PROMPT = """
    You are evaluating a student's answer to a bridge inspection question with emphasis on PRECISION and EFFICIENCY.
    You will be shown:
    1. The inspection question being asked
    2. Reference images that may be relevant (identified during QA generation, not definitive ground truth)
    3. Images the student selected to answer the question
    4. The student's text answer

    Your task is to evaluate the quality and accuracy of the student's answer based on ALL available visual evidence.

    IMPORTANT: The "reference images" are NOT definitive ground truth - they are examples identified as
    potentially relevant during QA generation. The student may have found OTHER relevant images not in
    the reference set. Judge the answer based on ALL images shown (both reference and student-selected).

    === EVALUATION PHILOSOPHY: PRECISION OVER RECALL ===

    PRIORITIZE students who select FEWER, MORE RELEVANT images over those who select MANY images with lower precision.

    Best practices in bridge inspection emphasize:
    - Identifying the MOST diagnostic images (quality over quantity)
    - Avoiding redundant or marginally relevant images (precision over recall)
    - Efficient documentation (2-5 highly relevant images typically sufficient)

    === SCORING RUBRIC: PRECISION-FOCUSED ===

    Evaluate on a CONTINUOUS scale from 0.0 to 1.0. Use the FULL range of scores:

    **HIGH SCORES (0.85-1.0): Excellent - Precision-Focused Selection**
    - Answer is factually accurate and well-supported
    - Student selected 2-5 highly relevant images that directly address the question
    - Minimal or no irrelevant images included
    - Demonstrates efficient, diagnostic image selection
    - May have selected DIFFERENT images than reference if equally/more relevant

    **GOOD SCORES (0.7-0.84): Good - Minor Inefficiencies**
    - Answer is accurate and supported by visual evidence
    - Student selected mostly relevant images but included 1-3 marginally relevant ones
    - Slight over-selection (1.5-2x reference count) but maintains good precision
    - Generally efficient with minor redundancy

    **MODERATE SCORES (0.5-0.69): Acceptable - Moderate Over-Selection**
    - Answer is partially correct with some inaccuracies or omissions
    - Student selected 2-3x more images than reference set
    - Many selected images are relevant, but significant noise/redundancy present
    - Demonstrates "cast wide net" strategy rather than diagnostic selection

    **LOW SCORES (0.3-0.49): Poor - Significant Issues**
    - Answer has significant inaccuracies or lacks proper visual support
    - Student selected 3x+ more images than reference set (severe over-selection)
    - OR selected mostly irrelevant images that don't address the question
    - Poor efficiency and diagnostic capability

    **VERY LOW SCORES (0.0-0.29): Failed - Critical Failures**
    - Answer contradicts visual evidence or is completely unsupported
    - Massive over-selection (5x+ images) suggesting unfocused analysis
    - OR critical images missing and answer is wrong

    === IMAGE SELECTION EVALUATION ===

    REWARD for:
    - Selecting 2-5 highly relevant images (optimal efficiency)
    - Finding critical diagnostic images (even if different from reference)
    - High per-image relevance (every image adds value)
    - Avoiding redundant images of the same defect/component

    PENALIZE for:
    - Selecting 2x+ more images than reference (start moderate penalty)
    - Selecting 3x+ more images than reference (strong penalty)
    - Selecting 5x+ more images than reference (severe penalty)
    - Including many marginally relevant or irrelevant images
    - Redundant images showing the same condition multiple times

    Do NOT penalize for:
    - Selecting different but equally relevant images than reference
    - Using slightly fewer images if they adequately support the answer
    - Different image filename references
    - Different wording if meaning is equivalent
    - Minor stylistic differences

    === EXAMPLES ===

    Example 1: Score 0.92 (Excellent Precision)
    - Reference: 4 images
    - Student: 3 images, all highly relevant, answer accurate
    - Reasoning: Efficient selection, high precision, good answer

    Example 2: Score 0.78 (Good with Minor Over-Selection)
    - Reference: 4 images
    - Student: 7 images, 6 relevant + 1 marginally relevant, answer accurate
    - Reasoning: Slight over-selection (1.75x) but maintains good precision

    Example 3: Score 0.58 (Moderate Over-Selection)
    - Reference: 4 images
    - Student: 12 images, found all critical ones but included 8 extras, answer mostly correct
    - Reasoning: 3x over-selection, "cast wide net" strategy, moderate precision

    Example 4: Score 0.35 (Poor - Severe Over-Selection)
    - Reference: 4 images
    - Student: 24 images, many irrelevant, answer partially correct
    - Reasoning: 6x over-selection, poor filtering, low precision

    === OUTPUT FORMAT ===

    CRITICAL: Return ONLY a decimal number between 0.0 and 1.0 (e.g., 0.85, 0.42, 0.93).
    Do NOT include any text, explanation, or additional output.
    Use precision to 2 decimal places (e.g., 0.75, not 0.7).
    Examples of valid responses: 0.85, 0.42, 0.93, 0.67
    Examples of INVALID responses: "0.7", "I rate this 0.85", "The score is 0.5"
"""

# Compute prompt hashes for cache invalidation
CORRECTNESS_PROMPT_HASH = hashlib.sha256(CORRECTNESS_PROMPT.encode()).hexdigest()
VLM_CORRECTNESS_PROMPT_HASH = hashlib.sha256(VLM_CORRECTNESS_PROMPT.encode()).hexdigest()


# =============================================================================
# Agent Factory Functions
# =============================================================================

def create_text_judge_agent(
    model: str = "litellm/gemini/gemini-3.1-flash-lite",
    prompt: str = CORRECTNESS_PROMPT
) -> Agent:
    """
    Create a text-based LLM judge agent for evaluating answer correctness.

    Args:
        model: LLM model identifier (default: gemini-2.5-flash)
        prompt: Evaluation prompt (default: CORRECTNESS_PROMPT)

    Returns:
        Configured Agent for text-based evaluation
    """
    return Agent(
        name="Evaluator",
        model=model,
        output_type=float,
        instructions=prompt
    )


def create_vlm_judge_agent(
    model: str = "litellm/gemini/gemini-3.1-flash-lite",
    prompt: str = VLM_CORRECTNESS_PROMPT
) -> Agent:
    """
    Create a vision-language model (VLM) judge agent for image-grounded evaluation.

    Args:
        model: VLM model identifier (default: gemini-2.5-flash)
        prompt: Evaluation prompt (default: VLM_CORRECTNESS_PROMPT)

    Returns:
        Configured Agent for VLM-based evaluation
    """
    return Agent(
        name="VLM_Evaluator",
        model=model,
        output_type=float,
        instructions=prompt
    )


# =============================================================================
# Global Cache Instances
# =============================================================================

_text_correctness_cache = TextCorrectnessCache(prompt_hash=CORRECTNESS_PROMPT_HASH)
_vlm_correctness_cache = VlmCorrectnessCache(prompt_hash=VLM_CORRECTNESS_PROMPT_HASH)

# Global agent instances (created lazily)
_text_judge_agent = None
_vlm_judge_agent = None


def _get_text_judge_agent() -> Agent:
    """Get or create the global text judge agent instance."""
    global _text_judge_agent
    if _text_judge_agent is None:
        _text_judge_agent = create_text_judge_agent()
    return _text_judge_agent


def _get_vlm_judge_agent() -> Agent:
    """Get or create the global VLM judge agent instance."""
    global _vlm_judge_agent
    if _vlm_judge_agent is None:
        _vlm_judge_agent = create_vlm_judge_agent()
    return _vlm_judge_agent


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_answer_correctness(
    answer: str,
    ground_truth: str,
    max_retries: int = 3,
    use_cache: bool = True,
) -> float:
    """
    Evaluate the correctness of an answer given the ground truth (synchronous).

    Args:
        answer: The student/agent answer to evaluate
        ground_truth: The reference/ground truth answer
        max_retries: Number of retries on LLM API failures (default: 3)
        use_cache: Whether to use cached results (default: True)

    Returns:
        Correctness score between 0.0 and 1.0
    """
    assert answer is not None, "Answer cannot be None"
    assert ground_truth is not None, "Ground truth cannot be None"

    # Check cache first
    if use_cache:
        cached_score = _text_correctness_cache.get(answer, ground_truth)
        if cached_score is not None and cached_score > 0:
            return cached_score

    template = f"""
    Now here is the student answer and the ground truth.

    Student Answer: {answer}\n
    Ground truth: {ground_truth}\n
    """

    agent = _get_text_judge_agent()

    for attempt in range(max_retries + 1):
        try:
            result = Runner.run_sync(agent, template).final_output
            # print(f"[LLM CALL] answer:{answer[:50]}... ground_truth:{ground_truth[:50]}... Correctness: {result:.3f}")

            # Store in cache
            if use_cache:
                _text_correctness_cache.set(answer, ground_truth, result)

            return result

        except Exception as e:
            print(
                f"[Retry {attempt}/{max_retries}] Error occurred with correctness metric: {e}"
            )
            continue

    # Return 0.0 on failure but don't cache it (might succeed on retry later)
    return 0.0


async def evaluate_answer_correctness_async(
    answer: str,
    ground_truth: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 8,
    use_cache: bool = True,
) -> float:
    """
    Async version of evaluate_answer_correctness with semaphore rate limiting.

    Args:
        answer: The student/agent answer to evaluate
        ground_truth: The reference/ground truth answer
        semaphore: Asyncio semaphore to limit concurrent requests
        max_retries: Number of retries on LLM API failures (default: 3)
        use_cache: Whether to use cached results (default: True)

    Returns:
        Correctness score between 0.0 and 1.0
    """
    assert answer is not None, "Answer cannot be None"
    assert ground_truth is not None, "Ground truth cannot be None"

    # Check cache first (cache hits don't need semaphore)
    if use_cache:
        cached_score = _text_correctness_cache.get(answer, ground_truth)
        if cached_score is not None:
            return cached_score

    # Acquire semaphore for LLM call
    async with semaphore:
        template = f"""
    Now here is the student answer and the ground truth.

    Student Answer: {answer}\n
    Ground truth: {ground_truth}\n
    """

        agent = _get_text_judge_agent()

        for attempt in range(max_retries + 1):
            try:
                result = await Runner.run(agent, template)
                score = result.final_output

                # Defensive: gemini-2.5-flash-lite sometimes returns 0-10 instead of 0-1.
                # If >1, treat as out-of-10 and scale down. Then clip to [0,1].
                if score is not None and score > 1.0:
                    score = score / 10.0
                if score is not None:
                    score = max(0.0, min(1.0, score))

                # Store in cache
                if use_cache:
                    _text_correctness_cache.set(answer, ground_truth, score)

                return score

            except Exception as e:
                print(
                    f"[Retry {attempt}/{max_retries}] Error occurred with correctness metric: {e}"
                )
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (capped)
                    await asyncio.sleep(min(2 ** attempt, 60))
                continue

        # Return 0.0 on failure but don't cache it
        return 0.0


async def evaluate_vlm_answer_correctness_async(
    answer: str,
    question: str,
    reference_images: List[str],
    scene_dir: str,
    base_scene_dir,
    semaphore: asyncio.Semaphore,
    agent_reference_images: List[str] | None = None,
    max_retries: int = 8,
    use_cache: bool = True,
) -> float:
    """
    Evaluate answer correctness using VLM with ground truth images.

    Shows the VLM:
    - Question being asked
    - Ground truth reference images
    - Which images the agent selected (if provided)
    - Agent's text answer

    Returns score 0.0-1.0 based on visual grounding and image selection quality.

    Args:
        answer: The agent's answer to evaluate
        question: The inspection question
        reference_images: List of ground truth image filenames
        scene_dir: Scene directory name
        base_scene_dir: Base directory containing scenes (Path object)
        semaphore: Asyncio semaphore to limit concurrent requests
        agent_reference_images: List of images the agent selected/referenced (optional)
        max_retries: Number of retries on LLM API failures (default: 3)
        use_cache: Whether to use cached results (default: True)

    Returns:
        Correctness score between 0.0 and 1.0
    """
    from .postprocessing import prepare_ground_truth_images

    assert answer is not None, "Answer cannot be None"
    assert question is not None, "Question cannot be None"

    # If no reference images, fall back to text-only evaluation
    if not reference_images:
        print(f"Warning: No reference images for VLM evaluation, skipping")
        return 0.0

    # Check cache first (cache hits don't need semaphore)
    if use_cache:
        cached_score = _vlm_correctness_cache.get(answer, question, reference_images)
        if cached_score is not None:
            return cached_score

    # Acquire semaphore for VLM call
    async with semaphore:
        # Load ground truth images
        try:
            image_dicts = prepare_ground_truth_images(reference_images, scene_dir, base_scene_dir)

            if not image_dicts:
                print(f"Warning: Could not load any images for VLM evaluation")
                return 0.0

            # Build prompt with question, images, and answer in OpenAI Agents SDK format
            content = [
                {"type": "input_text", "text": f"Question: {question}\n\nReference Images ({len(image_dicts)} images - identified as potentially relevant):"}
            ]
            content.extend(image_dicts)

            # Add agent's selected images (visually show them to VLM)
            if agent_reference_images:
                agent_img_count = len(agent_reference_images)
                content.append({
                    "type": "input_text",
                    "text": f"\n\nStudent Selected Images ({agent_img_count} images):"
                })

                # Load and show the agent's selected images
                agent_image_dicts = prepare_ground_truth_images(agent_reference_images, scene_dir, base_scene_dir)
                if agent_image_dicts:
                    content.extend(agent_image_dicts)
                else:
                    content.append({
                        "type": "input_text",
                        "text": f"(Could not load student's selected images: {', '.join(agent_reference_images)})"
                    })

            content.append({"type": "input_text", "text": f"\n\nStudent Answer: {answer}"})

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            agent = _get_vlm_judge_agent()

            for attempt in range(max_retries + 1):
                try:
                    result = await Runner.run(agent, messages)
                    raw_score = result.final_output

                    # Validate score to [0.0, 1.0]
                    if not isinstance(raw_score, (int, float)):
                        print(f"[VLM Retry {attempt}/{max_retries}] Invalid score type: {type(raw_score)}, got {raw_score}")
                        if attempt < max_retries:
                            await asyncio.sleep(1)
                        continue

                    score = float(raw_score)
                    if score < 0.0 or score > 1.0:
                        print(f"[VLM Retry {attempt}/{max_retries}] Score {score:.3f} out of range [0.0, 1.0]")
                        if attempt < max_retries:
                            await asyncio.sleep(1)
                        continue

                    # Valid score
                    # print(f"[VLM CALL] question:{question[:40]}... {len(image_dicts)} images, answer:{answer[:40]}... Score: {score:.3f}")

                    # Store in cache
                    if use_cache:
                        _vlm_correctness_cache.set(answer, question, reference_images, score)

                    return score

                except Exception as e:
                    print(f"[VLM Retry {attempt}/{max_retries}] Error: {e}")
                    if attempt < max_retries:
                        # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (capped)
                        await asyncio.sleep(min(2 ** attempt, 60))
                    continue

        except Exception as e:
            print(f"[VLM Error] Failed to prepare images: {e}")
            return 0.0

        # Return 0.0 on failure but don't cache it
        return 0.0
