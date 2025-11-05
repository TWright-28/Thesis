#!/usr/bin/env python3
"""
Local Bug Classification with GPT-OSS-20B via Ollama API
Simple, fast, and easy to use on your RTX 4090
"""

import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import argparse


class HarmonyResponseParser:
    """Parser for Harmony response format"""
    
    @staticmethod
    def extract_label(text: str) -> Optional[str]:
        """Extract classification label from text"""
        # Look for "Final Answer: <label>"
        pattern = r'Final\s+Answer\s*:\s*(Intrinsic|Extrinsic|Not\s+a\s+Bug|Unknown)'
        
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            if re.match(r'not\s+a\s+bug', label, re.IGNORECASE):
                return "Not a Bug"
            return label.title()
        
        return None
    
    @staticmethod
    def extract_reasoning(text: str) -> str:
        """Extract reasoning from response (everything before Final Answer)"""
        # Try to find content before "Final Answer:"
        match = re.search(r'(.+?)Final\s+Answer\s*:', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""


def call_ollama(prompt: str, model: str = "gpt-oss:20b", temperature: float = 0.2, max_tokens: int = 8192) -> str:

    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result.get("response", "")


def format_issue_prompt(issue: Dict[str, Any], fewshot_examples: str, reasoning_level: str = "high") -> str:
    """Format issue into classification prompt with all available metadata"""
    project = f"{issue['owner']}/{issue['repo']}"
    title = issue.get('title', 'No title')
    body = issue.get('body', 'No description')
    
    # Extract author information
    author_info = issue.get('author', {})
    author_name = author_info.get('username', 'Unknown') if author_info else 'Unknown'
    author_role = author_info.get('author_association', 'NONE') if author_info else 'NONE'
    
    # Extract labels
    labels = issue.get('labels', [])
    label_names = [label.get('name', '') for label in labels if label.get('name')]
    labels_str = ', '.join(label_names) if label_names else 'None'
    
    # Extract closing information
    closed_by = issue.get('closed_by')
    closing_pr = issue.get('closing_pr')
    closing_commit = issue.get('closing_commit')
    
    resolution_info = []
    if closed_by:
        resolution_info.append(f"Closed by: {closed_by}")
    if closing_pr:
        resolution_info.append(f"Fixed in PR: {closing_pr}")
    if closing_commit:
        resolution_info.append(f"Fixed in commit: {closing_commit}")
    resolution_str = '; '.join(resolution_info) if resolution_info else 'Unknown'
    
    # Format comments with author roles
    comments_list = issue.get('comments', [])
    formatted_comments = []
    
    for i, comment in enumerate(comments_list, 1):
        comment_author = comment.get('author', {})
        comment_user = comment_author.get('username', 'Unknown') if comment_author else 'Unknown'
        comment_role = comment_author.get('author_association', 'NONE') if comment_author else 'NONE'
        comment_body = comment.get('body', '')
        
        # Show role indicator for important roles
        role_indicator = ''
        if comment_role in ['OWNER', 'MEMBER', 'COLLABORATOR']:
            role_indicator = f' [{comment_role}]'
        
        formatted_comments.append(f"Comment {i} by {comment_user}{role_indicator}:\n{comment_body}")
    
    comments_str = '\n\n'.join(formatted_comments) if formatted_comments else 'No comments'
    
    # Truncate if extremely long
    if len(body) > 50000:
        body = body[:50000] + "\n\n[... truncated ...]"
    
    if len(comments_str) > 80000:
        comments_str = comments_str[:80000] + "\n\n[... truncated ...]"
    
    # Build the user task with enhanced metadata
    user_task = f"""

### Classification Task
**Bug Project:** {project}
**Bug Title:** {title}
**Issue Author:** {author_name} (Role: {author_role})
**Labels:** {labels_str}
**Resolution:** {resolution_str}

**Bug Description:**
{body}

**Discussion & Comments:**
{comments_str}

Follow the reasoning format above. Pay special attention to:
- Comments from maintainers (OWNER/MEMBER/COLLABORATOR) which may reveal root cause analysis
- Labels that indicate the nature of the issue  
- Resolution context (how it was fixed) which indicates whether the issue was in this repo or external
- Temporal phrases like "after upgrade", "since version X" that suggest external changes

Provide step-by-step analysis and end with:
Final Answer: <Intrinsic/Extrinsic/Not a Bug/Unknown>

Reasoning level: {reasoning_level}
"""
    
    # Combine: few-shot examples + user task
    full_prompt = fewshot_examples + user_task
    
    return full_prompt


def classify_issue(
    issue: Dict[str, Any],
    fewshot_examples: str,
    model: str = "gpt-oss:20b",
    reasoning_level: str = "high"
) -> Dict[str, Any]:
    """Classify a single issue using Ollama"""
    
    start_time = time.time()
    
    # Format the prompt
    prompt = format_issue_prompt(issue, fewshot_examples, reasoning_level)
    
    # Call Ollama
    print(f"  Generating (may take 30-90 seconds)...", end="", flush=True)
    
    try:
        response = call_ollama(prompt, model=model, temperature=0.2, max_tokens=8192)
    except Exception as e:
        print(f" ERROR: {e}")
        return None
    
    # Parse response
    parser = HarmonyResponseParser()
    predicted_label = parser.extract_label(response)
    reasoning = parser.extract_reasoning(response)
    
    if not predicted_label:
        predicted_label = "Unknown"
    
    # Get ground truth
    ground_truth = issue.get('final_classification', 'Unknown')
    
    inference_time = time.time() - start_time
    
    print(f" Done! ({inference_time:.1f}s)")
    
    return {
        "issue_number": issue.get('number'),
        "project": f"{issue['owner']}/{issue['repo']}",
        "title": issue.get('title', ''),
        "predicted_label": predicted_label,
        "ground_truth": ground_truth,
        "correct": predicted_label.lower() == ground_truth.lower(),
        "reasoning": reasoning,
        "full_response": response,
        "inference_time_seconds": round(inference_time, 2),
        "timestamp": datetime.now().isoformat(),
        "model": model
    }


def check_ollama_available(model: str = "gpt-oss:20b"):
    """Check if Ollama is running and model is available"""
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        
        # Check if model is available
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        if model not in model_names:
            print(f"⚠️  Model '{model}' not found in Ollama")
            print(f"Available models: {', '.join(model_names)}")
            print(f"\nTo install GPT-OSS-20B:")
            print(f"  ollama pull gpt-oss:20b")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        print("\nTo start Ollama:")
        print("  ollama serve")
        return False


def main():
    parser = argparse.ArgumentParser(description='Bug classification with Ollama API')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--prompt', type=str, required=True, help='Few-shot prompt file')
    parser.add_argument('--model', type=str, default='gpt-oss:20b', 
                       help='Ollama model name (default: gpt-oss:20b)')
    parser.add_argument('--reasoning', type=str, default='high', 
                       choices=['low', 'medium', 'high'])
    parser.add_argument('--limit', type=int, default=None, help='Limit number of issues')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Bug Classification with Ollama API")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Prompt: {args.prompt}")
    print(f"Reasoning: {args.reasoning}")
    if args.limit:
        print(f"Limit: {args.limit} issues")
    print("="*80)
    print()
    
    # Check Ollama availability
    print("Checking Ollama...")
    if not check_ollama_available(args.model):
        return
    print("✓ Ollama is running and model is available")
    print()
    
    # Load few-shot prompt
    print("Loading few-shot prompt...")
    with open(args.prompt, 'r', encoding='utf-8') as f:
        fewshot_examples = f.read().strip()
    print(f"Loaded ({len(fewshot_examples)} characters)")
    print()
    
    # Load issues
    print(f"Loading issues from {args.input}...")
    issues = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            issues.append(json.loads(line))
    
    if args.limit:
        issues = issues[:args.limit]
    
    print(f"Loaded {len(issues)} issues")
    print()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Process issues
    results = []
    correct_count = 0
    
    print("Starting classification...")
    print("-" * 80)
    
    for i, issue in enumerate(issues, 1):
        print(f"[{i}/{len(issues)}] Issue #{issue.get('number')} ({issue['owner']}/{issue['repo']})")
        
        try:
            result = classify_issue(issue, fewshot_examples, args.model, args.reasoning)
            
            if result is None:
                print("  Skipping due to error")
                continue
            
            results.append(result)
            
            if result['correct']:
                correct_count += 1
            
            # Print result
            print(f"  Predicted: {result['predicted_label']}")
            print(f"  Ground Truth: {result['ground_truth']}")
            print(f"  ✓ Correct" if result['correct'] else "  ✗ Incorrect")
            print()
            
            # Save incrementally
            with open(args.output, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving results...")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            import traceback
            traceback.print_exc()
            continue
    
    # Final statistics
    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Total processed: {len(results)}")
    print(f"Correct: {correct_count}")
    if results:
        print(f"Accuracy: {100 * correct_count / len(results):.2f}%")
        
        avg_time = sum(r['inference_time_seconds'] for r in results) / len(results)
        total_time = sum(r['inference_time_seconds'] for r in results)
        print(f"Average time: {avg_time:.1f}s per issue")
        print(f"Total time: {total_time/60:.1f} minutes")
    
    print(f"Results saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
