"use client";

import { cn } from "@/frontend/lib/utils";

interface MetricBarProps {
  label: string;
  value: number;
  maxValue?: number;
  color?: string;
  showPercentage?: boolean;
  className?: string;
}

export function MetricBar({
  label,
  value,
  maxValue = 1,
  color = "bg-primary",
  showPercentage = true,
  className,
}: MetricBarProps) {
  const percentage = Math.min((value / maxValue) * 100, 100);

  return (
    <div className={cn("space-y-1.5", className)}>
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">{label}</span>
        {showPercentage && (
          <span className="text-sm font-medium text-foreground">
            {(value * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <div className="h-2.5 w-full rounded-full bg-muted overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-700 ease-out", color)}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
