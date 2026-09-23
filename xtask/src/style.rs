//! The prose rules for markdown in this repository, checked rather than remembered.
//!
//! Three rules, all of them from CONTRIBUTING.md, all of them things a reviewer would otherwise
//! have to notice by eye on every pull request.
//!
//! No em dash and no en dash. They are used to join two clauses that a full stop or a comma joins
//! just as well, and a specification that uses them reads like it was generated rather than
//! written.
//!
//! No horizontal rules. A page break in a document that nobody prints is a break in the reading
//! and nothing else.
//!
//! No sentence broken across two lines. A paragraph is one physical line, however long. This is
//! the rule that makes a diff on a specification readable: changing a sentence changes one line
//! rather than reflowing a paragraph, so a review shows what was said differently instead of
//! showing where the wrapping moved.

use std::path::{Path, PathBuf};

pub(crate) fn check(root: &Path) -> Result<(), String> {
    let mut files = Vec::new();
    collect(root, &mut files)?;
    files.sort();

    let mut problems = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file)
            .map_err(|e| format!("could not read {}: {e}", file.display()))?;
        let shown = file.strip_prefix(root).unwrap_or(file).display().to_string();
        problems.extend(check_one(&shown, &text));
    }

    if problems.is_empty() {
        println!("the prose rules hold across {} markdown files", files.len());
        Ok(())
    } else {
        for problem in &problems {
            eprintln!("  {problem}");
        }
        Err(format!("{} prose violations", problems.len()))
    }
}

fn check_one(name: &str, text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut problems = Vec::new();
    let mut in_code = false;

    for (i, line) in lines.iter().enumerate() {
        let number = i + 1;
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        if let Some(column) = line.find(['\u{2014}', '\u{2013}']) {
            problems.push(format!("{name}:{number}:{column}: an em or en dash"));
        }
        let trimmed = line.trim();
        if matches!(trimmed, "---" | "***" | "___" | "- - -" | "* * *") {
            problems.push(format!("{name}:{number}: a horizontal rule"));
        }
        if is_prose(line) && continues(lines.get(i + 1).copied()) {
            problems.push(format!("{name}:{number}: a sentence broken across two lines"));
        }
    }
    problems
}

/// A line that is ordinary paragraph text, as opposed to a heading, a list item, a table row, an
/// indented block or a link reference. Only those get the one-line-per-paragraph rule, because the
/// others are structure and wrap for reasons of their own.
fn is_prose(line: &str) -> bool {
    if line.is_empty() || line.starts_with(' ') || line.starts_with('\t') {
        return false;
    }
    let first = line.chars().next().unwrap_or(' ');
    !matches!(first, '#' | '|' | '>' | '-' | '*' | '+' | '[' | '!' | '<')
        && !line.starts_with("1.")
        && !line.ends_with("  ")
}

/// The next line looks like the rest of the sentence above it: not blank, not structure, and
/// starting with something that cannot start a sentence.
fn continues(next: Option<&str>) -> bool {
    let Some(next) = next else { return false };
    if !is_prose(next) {
        return false;
    }
    let first = next.chars().next().unwrap_or(' ');
    first.is_lowercase() || first == ',' || first == ')'
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("could not read {}: {e}", dir.display()))?;
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        if path.is_dir() {
            collect(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_one;

    #[test]
    fn an_em_dash_is_caught() {
        assert_eq!(check_one("t.md", "A sentence \u{2014} and another.").len(), 1);
    }

    #[test]
    fn a_horizontal_rule_is_caught() {
        assert_eq!(check_one("t.md", "Above.\n\n---\n\nBelow.").len(), 1);
    }

    #[test]
    fn a_wrapped_sentence_is_caught() {
        assert_eq!(
            check_one("t.md", "The first half of a sentence\nand the second half.").len(),
            1
        );
    }

    #[test]
    fn two_paragraphs_are_fine() {
        assert!(check_one("t.md", "One paragraph.\n\nAnother paragraph.").is_empty());
    }

    #[test]
    fn structure_is_left_alone() {
        let text = "# Heading\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n- a list item\n- and another\n";
        assert!(check_one("t.md", text).is_empty());
    }

    #[test]
    fn a_dash_inside_a_code_fence_is_fine() {
        assert!(check_one("t.md", "Text.\n\n```\na \u{2014} b\n```\n").is_empty());
    }
}
