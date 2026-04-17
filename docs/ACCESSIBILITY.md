# Accessibility

The Bloc 4 brief requires the AI solution to be accessible to people with disabilities. Most of the accessibility work lives in the frontend (`mezonya.com`), but the API has to cooperate. This doc is the contract.

## What the API provides

**1. A plain-text reason for every prediction.**
The `reason` field is short, decodable out of context, and contains no HTML. The frontend renders it as the `aria-label` of the result card. Screen readers read:

> "Aqara and Philips work together with limitations. Shared protocols: zigbee, wifi."

instead of the silent `<span class="badge badge-partial"></span>` the visual markup would otherwise produce.

**2. A discrete label, not just a color.**
`compatibility` is `"compatible" | "partial" | "incompatible"`. The frontend must not rely on a green/yellow/red gradient alone, this fails WCAG 1.4.1 (Use of Color). Combine color with an icon and the label text.

**3. Self-descriptive OpenAPI spec.**
`/docs` (Swagger UI) and `/redoc` are generated from the same pydantic models the API validates against. They are keyboard-navigable and announce field descriptions to screen readers.

## What the frontend team needs to do

Minimum checklist for the compatibility quiz page:

- [ ] Every result card has `role="listitem"` and an `aria-label` built from `reason`.
- [ ] The 3-state compatibility badge uses color + icon + text, never color alone.
- [ ] Keyboard users can cycle through results with Tab; Enter opens the product page.
- [ ] Form inputs (ecosystem picker, category dropdown) have `<label for="">` associations.
- [ ] Confidence percentages are announced as "confidence 87 percent", not just "87".
- [ ] Loading states have `aria-live="polite"` so the SR announces "loading compatible devices".
- [ ] Error states have `role="alert"` and a useful message, not "Error 503".
- [ ] Page renders correctly at 200% zoom (WCAG 1.4.4).
- [ ] Contrast ratio >= 4.5:1 for body text, >= 3:1 for the badges.

## Testing

Manual: NVDA on Windows, VoiceOver on macOS, TalkBack on Android. Run the compatibility quiz end to end, eyes closed for the last third.

Automated: axe-core in the Playwright E2E suite. Fail the build on any WCAG A/AA violation.

## Non-goals

- The API is not localized. Text comes back in English. Localizing to French/Arabic is a frontend concern; the frontend keeps a `{label: translation}` map keyed on the discrete `compatibility` value.
- The API does not return audio, haptics, or Braille output. If we need them later they're transforms over `reason`.
