"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/frontend/components/ui/card";
import { Skeleton } from "@/frontend/components/ui/skeleton";

interface ModelComparisonChartProps {
  data?: { model: string; accuracy: number; f1Score: number; precision: number; recall: number }[];
}

export function ModelComparisonChart({ data }: ModelComparisonChartProps) {
  if (!data || data.length === 0) {
    return (
      <Card className="border-border/50 bg-card/50">
        <CardHeader>
          <CardTitle className="text-foreground">Model Comparison</CardTitle>
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
        <CardTitle className="text-foreground">Model Performance Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-2 text-muted-foreground font-medium">Model</th>
                <th className="text-right py-3 px-2 text-muted-foreground font-medium">Accuracy</th>
                <th className="text-right py-3 px-2 text-muted-foreground font-medium">F1 Score</th>
                <th className="text-right py-3 px-2 text-muted-foreground font-medium">Precision</th>
                <th className="text-right py-3 px-2 text-muted-foreground font-medium">Recall</th>
              </tr>
            </thead>
            <tbody>
              {data.map((item) => (
                <tr key={item.model} className="border-b border-border/50 hover:bg-muted/50 transition-colors">
                  <td className="py-3 px-2 text-foreground font-medium">{item.model}</td>
                  <td className="py-3 px-2 text-right text-foreground">{(item.accuracy * 100).toFixed(1)}%</td>
                  <td className="py-3 px-2 text-right text-foreground">{(item.f1Score * 100).toFixed(1)}%</td>
                  <td className="py-3 px-2 text-right text-foreground">{(item.precision * 100).toFixed(1)}%</td>
                  <td className="py-3 px-2 text-right text-foreground">{(item.recall * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
