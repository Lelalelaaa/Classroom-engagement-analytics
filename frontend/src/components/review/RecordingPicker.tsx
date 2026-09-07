'use client';
import { useRef } from 'react';
import { FileVideo } from 'lucide-react';

interface Props {
  fileName?: string;
  onPick: (file: File) => void;
}

export default function RecordingPicker({ fileName, onPick }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="flex items-center gap-3 flex-wrap">
      <button
        onClick={() => inputRef.current?.click()}
        className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-medium transition-colors"
      >
        <FileVideo className="w-4 h-4" />
        {fileName ? 'Choose a different file' : 'Choose video file'}
      </button>
      {fileName && <span className="text-sm text-gray-400 truncate max-w-xs">{fileName}</span>}
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onPick(file);
        }}
      />
    </div>
  );
}
