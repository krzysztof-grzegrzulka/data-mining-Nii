import re
import emoji


def sanitize_text(source_text: str, mentions: int = 0) -> str:
    if mentions == 1:
        output = re.sub(r'[!\"#$%&()*+\-.\\/:;<=>?@\[\]^_`{|}~\n]+',
                        '', source_text.lower())
    else:
        output = re.sub(r'@[\w\d]+', '', source_text.lower())
        output = re.sub(r'[!\"#$%&()*+\-.\\/:;<=>?@\[\]^_`{|}~\n]+',
                        '', output)
    output = output.encode('unicode-escape').decode('ASCII')
    output = re.sub(r'[!\"#$%&()*+\-.\\/:;<=>?@\[\]^_`{|}~\n]+', '', output)
    output = re.sub(r'[)(<>]+', '', output)
    output = re.sub(r'\d', '', output)
    output = re.sub(r'\\n', '', output)
    output = re.sub(r'"', '', output)
    output = re.sub(r'\(view spoiler\)\[', '', output)
    output = re.sub(r'\(hide spoiler\)]', '', output)
    output = re.sub(r'<\w*\s*\s*/*\w*[=\"]*>', '', output)
    output = re.sub(r'[,.;]', '', output)
    output = output.strip()
    output = re.sub(r"\s+", ' ', output)
    output = re.sub(r'[!\"#$%&()*+\-.\\/:;<=>?@\[\]^_`{|}~\n]+', '', output)

    return output
