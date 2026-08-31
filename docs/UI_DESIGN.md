# Interface design

The interface uses one light appearance, readable hierarchy, predictable
navigation, and immediate feedback. It keeps
Streamlit's native controls, keyboard behavior, and routing.
The simulation, models, governance gates, and committed results are unchanged.

- `app/styles.css` owns system typography, spacing, surfaces, control feedback,
  and accessibility media queries. Styles use Streamlit test IDs rather than
  generated CSS class names. Recheck these hooks when upgrading Streamlit.
- `app/common.py` owns grouped navigation. `app/ui.py` supplies reusable
  `page_header`, responsive `columns`, `Metric` / `metric_grid`, `card`, `badge`,
  `progress_steps`, and `plot` components. All pages use shared headers and rows;
  all Plotly charts use the shared renderer. Blue identifies
  navigation/current defense; red, green, and amber communicate attack, success,
  and caution. Status badges always include words, not just color.
- All application icons use Google Material Symbols Rounded, bundled locally
  with Streamlit. Native icon slots and Markdown tokens cover page icons,
  navigation, tabs, buttons, and messages; escaped HTML helpers cover cards,
  badges, and progress. CSS replaces framework utility artwork while retaining
  its original buttons, tooltips, and actions. Active navigation has a blue
  background, inset marker, and filled icon. Label font weight stays consistent
  between states so long labels do not rewrap when selected.
- The app is light-only: a pale canvas, white cards, and opaque navigation.
  `.streamlit/config.toml` sets a light theme and a minimal toolbar, removing
  the viewer theme selector. Plotly figures explicitly use `plotly_white` and
  `theme=None`, so chart surfaces do not follow an alternative viewer theme.
  No external font, script, animation library, or network request is added.
- Rows respond to their actual available width, including nested columns and
  the space left beside the sidebar. Metric grids align label/value/note rows
  using CSS subgrid with a flex fallback. Native metrics use content height,
  wrap long values, and never extend over following captions. Navigation link
  wrappers reset Streamlit's negative margins so their hit areas do not overlap.
- Wide tables and code keep their own scrolling rather than overflowing the
  page. Tabs and general progress steps wrap into rows, and narrow metric grids
  stack into one column. Sidebar labels wrap without truncation. The sidebar is
  permanently expanded, with its collapse/expand controls and empty header row
  removed. Below 900px it occupies a bounded, scrollable area above the page,
  keeping both navigation and content reachable without an overlay.
- Press feedback is immediate. There are no new drag gestures, artificial waits,
  input locks, or decorative entrance animations. Reduced motion removes press
  scaling and the brief opacity-only view transition. The sidebar width and
  scrollbar space stay stable across navigation; style injection and framework
  toolbar changes do not add empty layout rows. A small local styling script
  retains the stylesheet in the document head across Streamlit page cleanup;
  it does not intercept routing or widget events. Surfaces are already opaque,
  and higher contrast strengthens
  surface boundaries. Native widget behavior remains Streamlit's responsibility.
- Judge Demo stages stay mounted in overlapping grid cells that reserve the
  tallest stage's natural height at each viewport width. A Streamlit fragment
  updates only the controls and stage state; Next and Back do not rerun the page,
  reload the sidebar, or invoke scroll APIs. Panels crossfade without changing
  document height. The fixed-size Back / Restart / stage count / Next row remains
  above the horizontal progress stack, which scrolls internally on narrow screens.
  Next remains in place, disabled, at the final stage. Inactive panels are hidden
  from focus and accessibility navigation. Restart retains the original welcome flow.
- The Atlas exposes search and common filters first. Coverage and specialist
  filters remain available through named disclosures. Search combines all words
  across names, categories, objectives, mechanisms, and observed signals, alongside
  the selected filters. Clear filters restores the full catalog and default sort.

## Verification

Run `python -m pytest tests/test_app_navigation.py tests/test_artifact_consistency.py -q`
for offline page rendering, combined search/filter behavior, reset recovery,
all eight Judge Demo stages (including Back and Restart), all six Unseen Attack
Demo beats, and consistency of the committed results.

Check every page at 1440px, 768px, 390px, and 320px. Measure sibling component bounds,
metric-content containment, and horizontal page overflow; also inspect screenshots
and open each tab and demo state. Verify navigation remains expanded without
collapse controls or a header spacer, including after resizing. On phones,
scroll both the navigation region and the page to confirm all content is reachable. Inspect keyboard focus
and reduced-motion/higher-contrast preferences when adding effects. These are UI
checks, not validation of the fraud model against real payment data.

For Judge Demo, physically click through all stages in both directions with the
navigation row visible. Compare the scroll container's offset and navigation /
progress bounds before and after each click: all should stay unchanged. Use real
pointer clicks for this check; locator click helpers may scroll buttons into the
center of the viewport themselves and invalidate the scroll comparison.
