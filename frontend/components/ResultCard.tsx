"use client";

import { Card, CardContent } from "@/frontend/components/ui/card";
import { AlertTriangle, AlertCircle, Info } from "lucide-react";

export type SeverityLevel = "Fatal" | "Serious Injury" | "Slight Injury";

interface ResultCardProps {
  severity?: SeverityLevel;
  confidence?: number;
  model?: string;
  isLoading?: boolean;
}

const severityConfig: Record<
  SeverityLevel,
  { color: string; bgColor: string; icon: React.ElementType; description: string }
> = {
  Fatal: {
    color: "text-red-400",
    bgColor: "bg-red-500/10 border-red-500/30",
    icon: AlertTriangle,
    description:
      "High probability of fatal outcome. Immediate emergency response required.",
  },
  "Serious Injury": {
    color: "text-amber-400",
    bgColor: "bg-amber-500/10 border-amber-500/30",
    icon: AlertCircle,
    description:
      "Significant risk of serious injuries. Medical attention likely needed.",
  },
  "Slight Injury": {
    color: "text-emerald-400",
    bgColor: "bg-emerald-500/10 border-emerald-500/30",
    icon: Info,
    description:
      "Low severity expected. Minor injuries or property damage likely.",
  },
};

export function ResultCard({ severity, confidence, model, isLoading }: ResultCardProps) {
  if (isLoading) {
    return (
      <Card className="border-border/50 bg-card/50">
        <CardContent className="p-6">
          <div className="flex flex-col items-center justify-center gap-4 py-8">
            <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary/30 border-t-primary" />
            <p className="text-sm text-muted-foreground">Analyzing accident parameters...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!severity) {
    return (
      <Card className="border-border/50 bg-card/50">
        <CardContent className="p-6">
          <div className="flex flex-col items-center justify-center gap-3 py-8">
            <div className="h-16 w-16 rounded-full bg-muted flex items-center justify-center">
              <AlertCircle className="h-8 w-8 text-muted-foreground" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-muted-foreground">No Prediction Yet</p>
              <p className="text-xs text-muted-foreground/70 mt-1">
                Fill in all accident parameters and click &ldquo;Predict Severity&rdquo; to see results.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const config = severityConfig[severity];
  const Icon = config.icon;

  return (
    <Card className={`border ${config.bgColor}`}>
      <CardContent className="p-6">
        <div className="flex flex-col items-center text-center gap-4">
          <div className="h-20 w-20 rounded-full bg-background/80 flex items-center justify-center amber-glow">
            <Icon className={`h-10 w-10 ${config.color}`} />
          </div>

          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
              Predicted Severity
            </p>
            <h3 className={`text-2xl font-bold ${config.color}`}>{severity}</h3>
          </div>

          <div className="flex items-center gap-4 text-sm">
            {confidence !== undefined && (
              <div>
                <span className="text-muted-foreground">Confidence: </span>
                <span className="font-semibold text-foreground">
                  {(confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
            {model && (
              <div>
                <span className="text-muted-foreground">Model: </span>
                <span className="font-semibold text-foreground">{model}</span>
              </div>
            )}
          </div>

          <p className="text-sm text-muted-foreground max-w-sm">{config.description}</p>
        </div>
      </CardContent>
    </Card>
  );
}
