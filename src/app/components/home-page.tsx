import { Button } from "@/app/components/ui/button";
import { Card, CardContent } from "@/app/components/ui/card";
import { Upload, ScanSearch, Code, Download, CheckCircle, Shield, Zap } from "lucide-react";
import { LatexPreview } from "@/app/components/latex-preview";
import { motion } from "motion/react";

interface HomePageProps {
  onNavigate: (page: string) => void;
  onNewConvert: () => void;
}

export function HomePage({ onNavigate, onNewConvert }: HomePageProps) {
  const steps = [
    { icon: Upload, title: "Upload image", description: "Drag & drop or select an image with handwritten math" },
    { icon: ScanSearch, title: "AI recognition", description: "Deep learning model recognizes mathematical symbols" },
    { icon: Code, title: "Get LaTeX", description: "Instantly receive clean, editable LaTeX code" },
    { icon: Download, title: "Edit & export", description: "Fine-tune the result and copy to your document" },
  ];

  const features = [
    { icon: CheckCircle, title: "Mini-CoMER", description: "Transformer-based architecture with attention refinement for accurate recognition" },
    { icon: Zap, title: "Instant Results", description: "Get LaTeX code in under a second with GPU-accelerated inference" },
    { icon: Shield, title: "Local Processing", description: "All processing happens on your machine — no data leaves your device" },
  ];

  return (
    <div className="min-h-screen pb-20 md:pb-0">
      {/* Hero Section */}
      <section className="container mx-auto px-4 lg:px-6 py-12 lg:py-20">
        <div className="max-w-4xl mx-auto text-center space-y-6">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-3xl sm:text-4xl lg:text-5xl font-semibold text-foreground leading-tight"
          >
            Convert handwritten math
            <br />
            <span className="text-primary">into LaTeX instantly</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto"
          >
            Upload an image of a handwritten equation, and our AI model will recognize it and produce clean LaTeX code
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex flex-col sm:flex-row gap-3 justify-center pt-2"
          >
            <Button
              size="lg"
              onClick={onNewConvert}
              className="bg-primary text-primary-foreground hover:bg-primary/90 shadow-md hover:shadow-lg transition-shadow"
            >
              <Upload className="mr-2 h-5 w-5" />
              Start Converting
            </Button>
            <Button
              size="lg"
              variant="outline"
              onClick={() => onNavigate("convert")}
            >
              Try Demo
            </Button>
          </motion.div>

          {/* Before / After Comparison */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="pt-8 lg:pt-12"
          >
            <Card className="overflow-hidden">
              <CardContent className="p-6 lg:p-8">
                <div className="grid md:grid-cols-2 gap-6">
                  {/* Before — Handwritten style */}
                  <div className="space-y-3">
                    <div className="text-sm font-medium text-muted-foreground">Input</div>
                    <div className="bg-muted rounded-lg p-6 lg:p-8 flex items-center justify-center min-h-[160px] border-2 border-dashed border-border">
                      <div
                        className="text-3xl lg:text-4xl"
                        style={{ fontFamily: "'Caveat', 'Patrick Hand', cursive" }}
                      >
                        x&sup2; + 2xy + y&sup2;
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">Handwritten formula image</p>
                  </div>

                  {/* After — Rendered KaTeX */}
                  <div className="space-y-3">
                    <div className="text-sm font-medium text-muted-foreground">Output</div>
                    <div className="bg-card rounded-lg p-6 lg:p-8 flex items-center justify-center min-h-[160px] border shadow-sm">
                      <LatexPreview latex="x ^ { 2 } + 2 x y + y ^ { 2 }" />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      <code className="font-mono bg-muted px-1.5 py-0.5 rounded text-[11px]">
                        x ^ {"{"} 2 {"}"} + 2 x y + y ^ {"{"} 2 {"}"}
                      </code>
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </section>

      {/* How It Works */}
      <section className="bg-card py-12 lg:py-16 border-y border-border">
        <div className="container mx-auto px-4 lg:px-6">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-2xl lg:text-3xl font-semibold text-center mb-8 lg:mb-12">
              How It Works
            </h2>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
              {steps.map((step, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  className="text-center space-y-3"
                >
                  <div className="flex items-center justify-center">
                    <div className="bg-primary/10 p-4 rounded-xl">
                      <step.icon className="h-7 w-7 text-primary" />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-primary">Step {index + 1}</div>
                    <h3 className="font-medium text-sm">{step.title}</h3>
                    <p className="text-xs text-muted-foreground leading-relaxed">{step.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="container mx-auto px-4 lg:px-6 py-12 lg:py-16">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl lg:text-3xl font-semibold text-center mb-8 lg:mb-12">
            Built for Accuracy
          </h2>
          <div className="grid md:grid-cols-3 gap-6 lg:gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
              >
                <Card className="h-full hover:shadow-md transition-shadow">
                  <CardContent className="p-6 space-y-3">
                    <div className="bg-accent/10 p-3 rounded-lg w-fit">
                      <feature.icon className="h-5 w-5 text-accent" />
                    </div>
                    <h3 className="font-medium text-sm">{feature.title}</h3>
                    <p className="text-xs text-muted-foreground leading-relaxed">{feature.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="container mx-auto px-4 lg:px-6 py-12 lg:py-20">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="max-w-2xl mx-auto text-center space-y-6"
        >
          <h2 className="text-2xl lg:text-3xl font-semibold">
            Ready to convert your math?
          </h2>
          <Button
            size="lg"
            onClick={onNewConvert}
            className="bg-primary text-primary-foreground hover:bg-primary/90 shadow-md"
          >
            Get Started Now
          </Button>
        </motion.div>
      </section>
    </div>
  );
}
