"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/frontend/components/ui/card";
import { Skeleton } from "@/frontend/components/ui/skeleton";

interface SeverityPieChartProps {
  data?: { name: string; value: number; color: string }[];
}

export function SeverityPieChart({ data }: SeverityPieChartProps) {
  if (!data || data.length === 0) {
    return (
      <Card className="border-border/50 bg-card/50">
        <CardHeader>
          <CardTitle className="text-foreground">Accident Severity Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64">
            <div className="text-center space-y-3">
              <Skeleton className="h-48 w-48 rounded-full mx-auto" />
              <Skeleton className="h-4 w-32 mx-auto" />
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader>
        <CardTitle className="text-foreground">Accident Severity Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center gap-4">
          {data.map((item) => (
            <div key={item.name} className="flex items-center gap-3 w-full">
              <div
                className="h-4 w-4 rounded-full shrink-0"
                style={{ backgroundColor: item.color }}
              />
              <span className="text-sm text-muted-foreground flex-1">{item.name}</span>
              <span className="text-sm font-medium text-foreground">{item.value}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
