import { clsx } from 'clsx';

interface Props {
  label:   string;
  value:   string | number;
  accent?: 'green' | 'yellow' | 'indigo' | 'red';
}

const ACCENT_CLASSES = {
  green:  'text-green-400',
  yellow: 'text-yellow-400',
  indigo: 'text-indigo-400',
  red:    'text-red-400',
};

export function KPICard({ label, value, accent }: Props) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 transition-card">
      <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">{label}</p>
      <p className={clsx(
        'text-3xl font-bold',
        accent ? ACCENT_CLASSES[accent] : 'text-white',
      )}>
        {value}
      </p>
    </div>
  );
}
