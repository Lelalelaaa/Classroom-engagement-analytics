'use client';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { SegmentRating, SegmentReview } from '@/lib/types';

interface Props {
  index: number;
  count: number;
  startS: number;
  endS: number;
  value: SegmentReview;
  onChange: (review: SegmentReview) => void;
  onPrev: () => void;
  onNext: () => void;
}

const OPTIONS: { value: SegmentRating; label: string; hint: string; active: string }[] = [
  { value: 'focused', label: 'Focused', hint: 'Most of the class was with me', active: 'bg-green-500/15 border-green-500 text-green-300' },
  { value: 'mixed', label: 'Mixed', hint: 'Some were, some were not', active: 'bg-yellow-500/15 border-yellow-500 text-yellow-300' },
  { value: 'off', label: 'Not focused', hint: 'Most of the class was elsewhere', active: 'bg-red-500/15 border-red-500 text-red-300' },
];

function clock(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60);
  const s = Math.floor(totalSeconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function SegmentRater({ index, count, startS, endS, value, onChange, onPrev, onNext }: Props) {
  return (
    <div className="space-y-5">
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-widest">
          Segment {index + 1} of {count}
        </p>
        <p className="text-lg font-semibold mt-1">
          Minutes {clock(startS)}–{clock(endS)}
        </p>
      </div>

      <div>
        <p className="text-sm text-gray-300 mb-3">How engaged was the class during this stretch?</p>
        <div className="grid gap-2">
          {OPTIONS.map((opt) => {
            const selected = value.rating === opt.value;
            return (
              <button
                key={opt.value}
                onClick={() => onChange({ ...value, rating: opt.value })}
                className={`text-left px-4 py-3 rounded-xl border transition-colors ${
                  selected ? opt.active : 'bg-gray-900 border-gray-800 hover:border-gray-700'
                }`}
              >
                <span className="font-medium">{opt.label}</span>
                <span className="block text-xs text-gray-500 mt-0.5">{opt.hint}</span>
              </button>
            );
          })}
        </div>
      </div>

      <label className="flex items-center gap-2 text-sm text-gray-400">
        <input
          type="checkbox"
          checked={value.confidence === 'unsure'}
          onChange={(e) => onChange({ ...value, confidence: e.target.checked ? 'unsure' : 'sure' })}
          className="accent-indigo-500 w-4 h-4"
        />
        I&apos;m not sure about this one
      </label>

      <div>
        <label className="text-xs text-gray-500 uppercase tracking-widest">Note (optional)</label>
        <textarea
          value={value.note}
          onChange={(e) => onChange({ ...value, note: e.target.value })}
          rows={2}
          placeholder="e.g. group activity started here"
          className="mt-1 w-full bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:border-indigo-500"
        />
      </div>

      <div className="flex items-center justify-between pt-1">
        <button
          onClick={onPrev}
          disabled={index === 0}
          className="flex items-center gap-1 px-3 py-2 text-sm text-gray-400 hover:text-white disabled:opacity-30 disabled:hover:text-gray-400 transition-colors"
        >
          <ChevronLeft className="w-4 h-4" /> Previous
        </button>
        <button
          onClick={onNext}
          disabled={index === count - 1}
          className="flex items-center gap-1 px-4 py-2 text-sm font-medium bg-gray-800 hover:bg-gray-700 rounded-lg disabled:opacity-30 transition-colors"
        >
          Next segment <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
