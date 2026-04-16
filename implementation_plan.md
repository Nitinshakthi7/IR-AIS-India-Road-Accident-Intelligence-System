# UI Evaluation & Debugging Plan

Adopt the role of UI Tester and Debugger to systematically evaluate the three core sections of the IR-AIS platform: **Dashboard**, **Model Playground**, and **Live Predictor**.

## Goals
- Ensure a premium, "Five Pathways" inspired aesthetic (artisanal, heritage, cinematic).
- Verify high contrast and readability (following recent dark/bold text fixes).
- Audit responsiveness and interactivity across all components.
- Identify and resolve any layout shifts or functional UI bugs.

## Proposed Strategy

### 1. Dashboard Audit (`/?tab=dashboard`)
- **Visuals**: Verify the "Geometry of Risk" hero section and stats cards for proper alignment and shadow consistency.
- **Charts**: Test Recharts tooltips and labels. Ensure the "Pedestrian Involvement" pie chart has clear separation and distinct colors.
- **Responsiveness**: Check how the grid scales from 4 columns to 1 column on smaller viewports.

### 2. Model Playground Audit (`/?tab=models`)
- **Metric Comparison**: Audit the Radar charts and Bar charts for metric overlapping.
- **Tables**: Ensure the "Detailed Metric Scorecard" handles long model names gracefully (truncation/wrapping).
- **Tab Switching**: Verify that switching between Classification, Regression, and Auxiliary models is smooth and preserves state where expected.

### 3. Live Predictor Audit (`/?tab=predictor`)
- **Form UX**: Evaluate the collapsible section interaction. Ensure selects and inputs have consistent "Five Pathways" styling (bold borders, shadows).
- **Result Panels**: Test the "Predicted Severity" badge and "Casualty Estimation" person icons for visual polish and clarity.
- **Edge Cases**: Verify the "Awaiting Parameters" empty state and any error handling UI.

## Verification Plan

### Automated/Subagent Testing
- Use the `browser_subagent` to capture screenshots and recordings of key interactions.
- Audit Console logs for any React warnings or layout shift errors.

### Manual Refinement
- Adjust CSS or JSX if any component lacks the "premium" feel or exhibits layout issues.
- Polish micro-animations (fades, slides, count-ups).
