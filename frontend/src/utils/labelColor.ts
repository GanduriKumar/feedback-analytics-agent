const GOOGLE_LABEL_CLASSES = [
  'text-google-blue-700',
  'text-google-green-700',
  'text-google-red-700',
  'text-google-yellow-700',
  'text-google-gray-800',
] as const;

function hashString(s: string): number {
  // simple, stable hash (djb2-ish)
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = (h * 33) ^ s.charCodeAt(i);
  }
  return h >>> 0;
}

/**
 * "Random-looking" but stable per label.
 * This prevents colors changing every render while still appearing random across labels.
 */
export function labelColorClass(label: string): string {
  const idx = hashString(label) % GOOGLE_LABEL_CLASSES.length;
  return GOOGLE_LABEL_CLASSES[idx];
}
