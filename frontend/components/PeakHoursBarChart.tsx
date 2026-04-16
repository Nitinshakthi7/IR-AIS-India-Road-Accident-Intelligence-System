"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/frontend/components/ui/card";
import { Skeleton } from "@/frontend/components/ui/skeleton";

interface PeakHoursBarChartProps {
  data?: { hour: string; count: number }[];
}

export function PeakHoursBarChart({ data }: PeakHoursBarChartProps) {
  if (!data || data.length === 0) {
    return (
      <Card className="border-border/50 bg-card/50">
        <CardHeader>
          <CardTitle className="text-foreground">Peak Accident Hours</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    );
  }

  const maxCount = Math.max(...data.map((d) => d.count));

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader>
        <CardTitle className="text-foreground">Peak Accident Hours</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-end gap-1.5 h-64 overflow-x-auto pb-2">
          {data.map((item) => {
            const height = (item.count / maxCount) * 100;
            return (
              <div key={item.hour} className="flex flex-col items-center gap-1 flex-1 min-w-[28px]">
                <span className="text-xs text-muted-foreground">{item.count}</span>
                <div
                  className="w-full rounded-t bg-primary/70 hover:bg-primary transition-colors min-h-[2px]"
                  style={{ height: `${height}%` }}
                />
                <span className="text-[10px] text-muted-foreground">{item.hour}</span>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
