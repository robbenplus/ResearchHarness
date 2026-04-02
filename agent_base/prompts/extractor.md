Please process the following webpage content and user goal to extract relevant information.

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content. Preserve the most useful original context as fully as practical.
3. **Summary Output for Summary**: Organize a concise, goal-focused summary with clear logical flow.

## **Output Requirements**
- Return a single JSON object only.
- Required keys: `"rational"`, `"evidence"`, `"summary"`.
- All three fields must always be present.
- `"evidence"` and `"summary"` must be non-empty strings whenever relevant content exists.
- If the page is irrelevant or insufficient, still return valid strings explaining that limitation.
