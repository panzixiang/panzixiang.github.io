# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based personal blog hosted on GitHub Pages, built on the Jekyll Now theme. The site is deployed at <https://www.panzixiang.com> (custom domain: seanpan.me).

## Development Commands

```bash
# Install dependencies (one-time setup)
gem install github-pages

# Serve locally with auto-reload
jekyll serve

# View at http://127.0.0.1:4000/
```

GitHub Pages automatically rebuilds on push to master branch.

## Content Architecture

### Post Categories and Routing

Posts are organized by category with case-sensitive filtering:

| Category | Directory | URL | Filter Condition |
| ---------- | ----------- | ----- | ------------------ |
| Projects | `_posts/Projects/` | `/` (homepage) | `post.category != 'Blog'` |
| Blog | `_posts/Blog/` | `/Blog` | `post.category == 'Blog'` |
| NLP | `_posts/NLP/` | `/NLP` | `post.category == 'nlp'` (lowercase!) |
| AI | `_posts/AI/` | `/AI` | `post.category == 'AI'` |

**Important:** The NLP category uses lowercase `nlp` in front matter despite the uppercase directory name.

Follow markdownlint linting rules when writing all .md files

### Creating Posts

Post filename format: `YYYY-MM-DD-Title.md`

Required front matter:

```yaml
---
layout: post
title: Article Title
category: AI  # Use: Blog, nlp, Projects, or AI
---
```

Use `<!--more-->` to define excerpt boundary for listing pages.

## Layout System

Three layouts in `_layouts/`:

- **default.html** - Master template with header, navigation, footer
- **post.html** - Blog posts (extends default, adds date and Disqus)
- **page.html** - Static pages (extends default, no date/comments)

Navigation order: Projects | Blog | NLP Notes | AI | About | Work

## Styling

Dark theme with key colors defined in `_sass/_variables.scss`:

- Background: `#282c34`
- Text: `#c3cee3`
- Accent (h1, buttons): `#ffcb6b`
- Links (h2-h4): `#82aaff`

Main stylesheet: `style.scss` imports partials from `_sass/`.

## Configuration

Key settings in `_config.yml`:

- Permalink structure: `/:categories/:title/`
- Markdown: GitHub Flavored Markdown (kramdown)
- Syntax highlighting: Rouge
- Plugins: jekyll-sitemap, jekyll-feed

## Monthly AI Landscape Post

Create a monthly "AI Landscape" post summarizing developments from the previous month. When the user requests this (e.g., "Write the AI Landscape post for February 2026"), follow these instructions:

### File Setup

- **Filename**: `_posts/AI/YYYY-MM-DD-claude{MM}{YY}.md`
- **Date**: Use the current date or end of month
- **Example**: `2026-02-28-claude0226.md` (for February 2026)

### Front Matter

```yaml
---
layout: post
title: AI Landscape - [Month] [Year]
category: AI
---
```

### Content Structure

Research and write about developments from the **previous month** in these sections:

1. **Claude Model Updates** - New Claude models, capabilities, pricing changes
2. **Claude Code Updates** - New features, improvements, milestones
3. **Claude API & Platform Updates** - API features, MCP updates, SDK changes
4. **Agentic AI Frameworks** - Popular frameworks (LangGraph, CrewAI, MetaGPT, etc.)
5. **Financial AI & LLM Projects** - Finance-focused AI tools and repos
6. **Developer Tools & Infrastructure** - LLM gateways, MCP servers, dev tools
7. **Industry Partnerships** - Major AI partnerships and announcements

### Research Process

1. Use WebSearch to find recent announcements and updates
2. Use WebFetch to verify all GitHub links are working
3. Include star counts for GitHub repos where available
4. Each entry should have 2-4 sentences of description
5. All links must be verified working before publishing

### Example Entry Format

```markdown
### [Project Name] ([star count] stars)

[2-4 sentence description of what it does, key features, and why it matters.]

[Link Text](https://verified-url.com)
```

### Footer

End each post with:

```markdown
---

Last updated: [Month] [Day], [Year]
```

### After Writing

1. Verify all hyperlinks work
2. Run markdownlint checks
3. Commit with message: "Add AI Landscape [Month] [Year] blog post"
4. Push to master
