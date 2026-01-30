import type { AnalysisReport, ClusterResponse, ThemeData, UserType, DataSource, LLMConfig } from '../types';

function normalizeText(s?: string | null) {
  return (s || '').trim();
}

export function normalizeThemeCategory(theme?: string | null): string {
  const raw = (theme || '').trim();
  if (!raw) return 'General';
  const cleaned = raw
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();

  if (!cleaned || cleaned === 'unknown' || cleaned === 'unclassified') return 'General';

  const mapping: Record<string, string> = {
    connectivity: 'Connectivity',
    bluetooth: 'Connectivity',
    wifi: 'Connectivity',
    network: 'Connectivity',
    battery: 'Battery',
    charging: 'Battery',
    power: 'Battery',
    camera: 'Camera',
    display: 'Display',
    screen: 'Display',
    performance: 'Performance',
    stability: 'Stability',
    crash: 'Stability',
    freeze: 'Stability',
    audio: 'Audio',
    speaker: 'Audio',
    mic: 'Audio',
    update: 'Update',
    pricing: 'Pricing',
    price: 'Pricing',
    cost: 'Pricing',
    design: 'Design',
    ux: 'UX',
    ui: 'UX',
    usability: 'UX',
    support: 'Support',
    'customer service': 'Support',
  };

  for (const key of Object.keys(mapping)) {
    if (cleaned.includes(key)) return mapping[key];
  }

  return cleaned
    .split(' ')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export function getUniqueThemeCount(themes: ThemeData[]): number {
  const unique = new Set<string>();
  themes.forEach((t) => unique.add(normalizeThemeCategory(t.theme)));
  return unique.size;
}

export function deriveReport(args: {
  userType: UserType;
  searchQueries: string[];
  sources: DataSource[];
  llmConfig: LLMConfig;
  totalReviews: number;
  themes: ThemeData[];
  clusters?: ClusterResponse | null;
}): AnalysisReport {
  const id = `report-${Date.now()}`;
  const generated_at = new Date().toISOString();

  const products = new Set<string>();
  const functionalities = new Set<string>();
  const issue_categories: Record<string, number> = {};
  const sentiment_distribution: Record<string, number> = {};

  for (const t of args.themes) {
    const p = normalizeText(t.product);
    if (p) products.add(p);

    const func = normalizeText(t.theme);
    if (func) functionalities.add(func);

    const cat = normalizeText(t.classification) || 'Unclassified';
    issue_categories[cat] = (issue_categories[cat] || 0) + 1;

    const s = normalizeText(t.sentiment) || 'unknown';
    sentiment_distribution[s] = (sentiment_distribution[s] || 0) + 1;
  }

  const avg_review_length = args.totalReviews > 0 ? 0 : 0; // filled by pipeline if we keep raw strings; placeholder

  const recommendations: string[] = [];
  const sortedCats = Object.entries(issue_categories).sort((a, b) => b[1] - a[1]).slice(0, 5);
  for (const [cat, count] of sortedCats) {
    const pct = args.themes.length ? Math.round((count / args.themes.length) * 100) : 0;
    recommendations.push(`Prioritize ${cat} (${pct}% of identified themes).`);
  }
  if (!recommendations.length) {
    recommendations.push('Run an analysis with more data to generate actionable recommendations.');
  }

  return {
    id,
    generated_at,
    user_type: args.userType,
    search_queries: args.searchQueries,
    data_sources: args.sources,
    llm_config: args.llmConfig,

    total_reviews: args.totalReviews,
    total_themes: args.themes.length,

    products: Array.from(products).sort(),
    functionalities: Array.from(functionalities).sort(),

    issue_categories,
    sentiment_distribution,

    themes: args.themes,
    clusters: args.clusters ?? null,

    recommendations,
    text_analytics: {
      avg_review_length,
      unique_products: products.size,
      unique_functionalities: functionalities.size,
      unique_issue_categories: Object.keys(issue_categories).length,
    },
  };
}
