import re


def sanitize_text(source_text: str) -> str:
    output = re.sub(r'[:;]{1}-?[)(<>]{1}', '', source_text.lower())
    output = re.sub(r'\d', '', output)
    output = re.sub(r'<\w*\s*\s*/*\w*[=\"]*>', '', output)
    output = re.sub(r'[,.;]', '', output)
    output = output.strip()
    output = re.sub(r"\s+", ' ', output)

    return output
