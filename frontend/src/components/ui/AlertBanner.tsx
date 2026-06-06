import { Alert } from '@/lib/types';
import { AlertTriangle } from 'lucide-react';

interface Props { alert: Alert }

export function AlertBanner({ alert }: Props) {
  return (
    <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3">
      <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" />
      <div>
        <p className="text-sm font-semibold text-red-400">Low Engagement Alert</p>
        <p className="text-xs text-gray-400">{alert.message}</p>
      </div>
      <div className="ml-auto">
        <span className="text-2xl font-bold text-red-400">{alert.score.toFixed(0)}%</span>
      </div>
    </div>
  );
}
