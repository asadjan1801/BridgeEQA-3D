"""
Prompt templates for Gemini PDF QA Generation Pipeline
"""

QA_GENERATION_INSTRUCTIONS = """
Generate atleast 20 distinct Q/A pairs from the bridge inspection report following these CRITICAL requirements:

**CRITICAL REQUIREMENT**: The 'answer' field MUST contain VERBATIM text from the inspection report - exact quotes only!

**PRIORITIZE MULTI-HOP REASONING QUESTIONS** (At least 70% of questions should be multi-hop):
Multi-hop questions require understanding relationships between multiple bridge components, comparative analysis, or causal reasoning. These questions test deeper comprehension rather than simple fact retrieval.

Types of Multi-Hop Questions to Generate:
1. **Comparative Analysis**: Compare conditions across multiple components or locations
   - "How do the conditions of Abutment 1 and Abutment 2 compare, and what are their respective impacts on the overall substructure rating?"

2. **Causal Reasoning**: Identify cause-effect relationships between different defects
   - "What deterioration patterns are observed in both the deck and superstructure that indicate water infiltration issues?"

3. **Component Interaction**: Understand how different components affect each other
   - "How does the bearing condition relate to the observed deterioration in the adjacent superstructure elements?"

4. **Progressive Assessment**: Track deterioration progression across multiple elements
   - "Based on the spalling patterns observed in the piers and abutments, what is the overall assessment of the substructure's concrete condition?"

5. **System-Level Analysis**: Synthesize information from multiple components for overall assessment
   - "Considering the conditions of the deck, joints, and bearings, what is the overall water management performance of the bridge?"

Guidelines for Questions:
1. Frame using qualitative descriptions of observable conditions
2. Test direct assessment skills requiring synthesis of multiple observations
3. Explicitly request ratings when you want numerical ratings in the answer
   - **IMPORTANT**: Only ask for ONE condition rating per question (e.g., ask for deck rating OR superstructure rating, not both)
4. NEVER reference photo numbers, figure numbers, or specific report sections
5. NEVER use Condition State codes (CS1, CS2, etc.) in questions
6. **MANDATORY**: Generate questions that require information from AT LEAST 2 different components/elements
7. Questions should require reasoning, not just concatenation of facts
8. **MANDATORY**: Every major component that has a condition rating must be involved in a question that also explicitly asks for a condition rating.

Guidelines for Answers:
1. MUST be EXACT quotes from the inspection report - copy text verbatim
2. For multi-hop questions, combine verbatim quotes from different sections as needed
3. Copy text exactly as it appears, including all technical terms and measurements
4. For multi-component questions, answers may combine passages from different report sections
5. Ensure the combined passages logically answer the multi-hop question
6. **INLINE IMAGE CITATIONS (MANDATORY)**: Every claim about visible damage, deterioration,
   or physical condition MUST include an inline citation in the format [img:filename.png]
   immediately after the claim. Use the exact image filename (hash.png) from the
   Image Reference Mappings. Example:
   "The deck underside exhibits map cracking with heavy efflorescence [img:dea63df69f504dc577cb28a729083e55.png]."
7. A single image may be cited multiple times if it supports multiple claims
8. Generic summary text (e.g., "Overall, the condition is satisfactory") does NOT need a citation
9. The reference_images field MUST contain the deduplicated list of ALL image filenames
   cited via [img:...] tokens in the answer text

Important Notes:
- Use extracted observations in 'condition_ratings' as verbatim text sources
- Each answer must be traceable to specific locations in the source document
- Multi-hop answers should maintain logical flow even when combining multiple quotes
- Focus on questions that would help train an AI to understand structural relationships

Example Multi-Hop Q&A Pairs:

Example 1 (Comparative):
Question: "Compare the deterioration levels between the north and south abutments and explain how they contribute to the overall substructure rating."
Answer: "Abutment 1 North has heavy spalling at seating and stem interface with exposed rebar visible [img:a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6.png]. Abutment 2 South has minor cracking along stem and debris buildup along seating with deep spalling along center section [img:f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3.png]. The extensive deterioration at both abutments resulted in a substructure rating of 4."

Example 2 (Causal):
Question: "What evidence of water damage is present across the deck and superstructure components, and what is the deck's condition rating?"
Answer: "The deck shows extensive delamination and spalling with efflorescence visible at multiple locations [img:1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d.png]. The superstructure exhibits rust staining and pack rust at beam ends where water has penetrated through deck joints [img:d6c5b4a3f2e1d6c5b4a3f2e1d6c5b4a3.png]. Deck rating is 5."

Example 3 (System-Level):
Question: "Based on the conditions observed in the bearings, expansion joints, and approach slabs, assess the bridge's movement accommodation system."
Answer: "Bearings show moderate corrosion with frozen plates preventing proper movement [img:7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d.png]. Expansion joints are filled with debris and show 2-inch separation [img:2d1c0b9a8f7e6d5c4b3a2f1e0d9c8b7a.png]. Approach slabs have settled 3 inches creating a bump at bridge entrance [img:3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b.png]. These movement system deficiencies contribute to accelerated deterioration."
"""