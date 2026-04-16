"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/frontend/components/ui/card";
import { Button } from "@/frontend/components/ui/button";
import { Label } from "@/frontend/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/frontend/components/ui/select";
import { ALL_FEATURE_OPTIONS, ALL_FEATURES, type FeatureOption } from "@/frontend/lib/feature-options";
import { Zap, RotateCcw } from "lucide-react";

export interface AccidentFormData {
  [key: string]: string;
}

interface AccidentFormProps {
  onSubmit?: (data: AccidentFormData) => void;
  isLoading?: boolean;
}

export function AccidentForm({ onSubmit, isLoading }: AccidentFormProps) {
  const [formData, setFormData] = useState<AccidentFormData>({});

  const handleChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit?.(formData);
  };

  const handleReset = () => {
    setFormData({});
  };

  const filledCount = Object.keys(formData).length;
  const totalFields = ALL_FEATURES.length;
  const canSubmit = filledCount === totalFields;

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-foreground">Accident Parameters</CardTitle>
          <span className="text-xs text-muted-foreground">
            {filledCount}/{totalFields} fields
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[500px] overflow-y-auto pr-1">
            {ALL_FEATURES.map((feature) => {
              const featureDef: FeatureOption | undefined =
                ALL_FEATURE_OPTIONS[feature];
              if (!featureDef) return null;
              return (
                <div key={feature} className="space-y-1.5">
                  <Label
                    htmlFor={feature}
                    className="text-xs text-muted-foreground"
                  >
                    {featureDef.label}
                  </Label>
                  <Select
                    value={formData[feature] || ""}
                    onValueChange={(val) => handleChange(feature, val)}
                  >
                    <SelectTrigger id={feature} className="bg-background border-border/50">
                      <SelectValue placeholder={`Select ${featureDef.label.toLowerCase()}`} />
                    </SelectTrigger>
                    <SelectContent className="bg-card border-border max-h-60 overflow-y-auto">
                      {featureDef.options.map((opt) => (
                        <SelectItem key={opt} value={opt}>
                          {opt}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              );
            })}
          </div>

          <div className="flex items-center gap-3 pt-2 border-t border-border/50">
            <Button
              type="submit"
              disabled={!canSubmit || isLoading}
              className="bg-primary text-primary-foreground hover:bg-primary/90"
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Predicting...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  Predict Severity
                </span>
              )}
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={handleReset}
              className="border-border text-muted-foreground hover:text-foreground hover:bg-muted"
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
