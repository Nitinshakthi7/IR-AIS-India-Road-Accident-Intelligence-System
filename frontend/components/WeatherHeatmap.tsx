"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/frontend/components/ui/card";
import { Skeleton } from "@/frontend/components/ui/skeleton";

interface WeatherHeatmapProps {
  data?: { weather: string; severity: string; count: number }[];
}

export function WeatherHeatmap({ data }: WeatherHeatmapProps) {
  if (!data || data.length === 0) {
    return (
      <Card className="border-border/50 bg-card/50">
        <CardHeader>
          <CardTitle className="text-foreground">Weather & Severity Heatmap</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader>
        <CardTitle className="text-foreground">Weather & Severity Heatmap</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {data.map((item, idx) => {
            const intensity = Math.min(item.count / 1000, 1);
            return (
              <div key={idx} className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground w-28 truncate">
                  {item.weather}
                </span>
                <div className="flex-1 h-8 rounded bg-muted overflow-hidden">
                  <div
                    className="h-full rounded bg-gradient-to-r from-primary/40 to-primary transition-all"
                    style={{ width: `${intensity * 100}%` }}
                  />
                </div>
                <span className="text-sm text-muted-foreground w-24 truncate text-right">
                  {item.severity}: {item.count}
                </span>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
