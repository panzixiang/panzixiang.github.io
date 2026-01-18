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
