# UI Redesign: Five Pathways–Inspired Aesthetic for IR-AIS

Transform the IR-AIS dashboard from its current dark navy/charcoal theme into a **warm, editorial, craft-inspired** design language inspired by [fivepathways.com](https://fivepathways.com).

## Design Language Translation

| Five Pathways Element | IR-AIS Adaptation |
|---|---|
| Warm off-white background (`#FFFaf5`) | Warm cream/paper background for the app |
| Serif headlines (PP Editorial New) | **Playfair Display** (free Google Font) for headings |
| Clean sans-serif body (Neue Montreal) | **Inter** for body/nav/data — already widely available |
| Thick black borders + hard offset shadow | Cards & buttons with `2px solid black` + `4px 4px 0 black` offset shadow |
| Teal accent (`#6bc4b3`) | Primary accent for CTAs and active states |
| Red accent (`#e05a47`) | Alert/danger accent for severity warnings |
| Pill-shaped buttons with arrows | Rounded pill buttons with line-drawn arrow icons |
| Generous whitespace & spacious layout | Increased card padding, generous section spacing |
| Hand-drawn/etched illustration style | Line-art SVG icons & cross-hatch textures on decorative elements |
| Hover "click-down" on buttons | Shadow removal + translate on `:active` for physical press feel |
| Paper-like texture/grain | Subtle CSS noise texture overlay on background |

## Proposed Changes

### Styling Foundation

#### [MODIFY] [globals.css](file:///c:/Users/tejas/Downloads/workspace-6e15fe32-0203-4fac-bd02-0309e523d20f/src/app/globals.css)
Complete theme overhaul:
- Replace dark oklch color system with warm cream palette (`#FFFaf5`, `#ebdec5`, `#1a1a1a`, `#6bc4b3`, `#e05a47`)
- Import Google Fonts: **Playfair Display** (serif) + **Inter** (sans-serif)
- New card styles: thick black borders, hard offset shadows, large rounded corners
- Button styles: pill-shaped with border, shadow, active press-down animation
- Subtle paper grain noise texture on `body`
- Updated scrollbar to match warm theme
- Adapted chart tooltip styles for light theme
- New animation keyframes for smooth entrance on scroll
- Tab pills: black border, teal active fill
- Form section styles adapted for light theme

---

### Typography & Layout (Root Layout)

#### [MODIFY] [layout.tsx](file:///c:/Users/tejas/Downloads/workspace-6e15fe32-0203-4fac-bd02-0309e523d20f/src/app/layout.tsx)
- Switch from Geist to **Playfair Display** + **Inter** via `next/font/google`
- Remove `dark` class from `<html>`
- Add noise texture class to body

---

### Navigation

#### [MODIFY] [Navbar.tsx](file:///c:/Users/tejas/Downloads/workspace-6e15fe32-0203-4fac-bd02-0309e523d20f/frontend/components/Navbar.tsx)
- Warm cream navbar with black text
- Logo with editorial serif styling
- Nav links: pill-shaped with thick borders, teal active state
- Top accent line: warm teal gradient instead of amber
- Mobile menu: cream card with soft shadow
- Add `→` arrow to active nav link

---

### Main Page & Dashboard

#### [MODIFY] [page.tsx](file:///c:/Users/tejas/Downloads/workspace-6e15fe32-0203-4fac-bd02-0309e523d20f/src/app/page.tsx)
- Hero section: editorial serif headline on cream background with decorative line-art
- Stat cards: warm backgrounds with thick borders and offset shadows
- Chart cards: cream backgrounds with thick black borders
- Chart themes: adapted for light background (dark grid lines, warm colors)
- Tooltip styles: light cream with black border + offset shadow
- Tab navigation: thick-bordered pill buttons with teal active state
- Tables: cream rows, thick header border
- Severity badges: warm accent colors with thick borders
- Buttons: pill-shaped with hard shadow + press-down interaction
- Footer: simple black text on cream

---

### Chart Color Adaptation

All recharts components switch from dark-mode colors to warm-palette equivalents:
- Grid lines: `#d4c5a9` (warm gray)
- Axis text: `#3d3d3d` (near-black)
- Severity colors: `#e05a47` (fatal), `#d4a843` (serious), `#6bc4b3` (slight)
- Bar fills: warm earth tones with teal/coral accents

## User Review Required

> [!IMPORTANT]
> This is a **complete visual overhaul** — every component moves from dark navy to a warm, editorial light theme. The layout structure and functionality remain identical.

> [!WARNING]
> The chart tooltip and axis styles are all inline in `page.tsx`. These will all be updated to match the warm light theme.

## Open Questions

1. **Do you also want to keep a dark mode option?** The inspiration site is light-only. I can make this light-only, or keep a dark mode toggle with adapted colors.
2. **Font licensing**: Playfair Display (Google Fonts) is the closest free match to the editorial serif. Are you ok with that, or do you have a preference?

## Verification Plan

### Automated Tests
- Fix the `npm run dev` script for Windows (`tee` → redirect)
- Run `npm run dev` and verify the app starts without errors
- Browser test to visually verify the new design

### Manual Verification
- Screenshot comparison against inspiration site aesthetic
- Check responsive layout on mobile vs desktop
- Verify all chart readability on light background
