"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { Menu, X, ShieldAlert, BarChart3, BrainCircuit, Zap, Home } from "lucide-react";
import { Button } from "@/frontend/components/ui/button";
import { cn } from "@/frontend/lib/utils";

const navLinks = [
  { href: "/?tab=home", label: "Home", hash: "home", icon: Home },
  { href: "/?tab=dashboard", label: "Dashboard", hash: "dashboard", icon: BarChart3 },
  { href: "/?tab=models", label: "Model Playground", hash: "models", icon: BrainCircuit },
  { href: "/?tab=predictor", label: "Live Predictor", hash: "predictor", icon: Zap },
];

export function Navbar() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const currentTab = searchParams.get("tab") || "home";

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleNav = (link: (typeof navLinks)[0]) => {
    router.push(link.href);
    setMobileOpen(false);
  };

  return (
    <>
      {/* Top teal accent line */}
      <div className="fixed top-0 left-0 right-0 h-[3px] z-[60] bg-primary shadow-[0_0_10px_rgba(107,196,179,0.3)]" />

      <header className={cn(
        "sticky top-[3px] z-50 w-full transition-all duration-300",
        scrolled
          ? "bg-background/95 border-b-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]"
          : "bg-background border-b border-black/10"
      )}>
        <div className="mx-auto flex h-18 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          {/* Logo - Editorial Serif */}
          <button onClick={() => handleNav(navLinks[0])} className="flex items-center gap-2 group cursor-pointer">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-black text-primary border-2 border-black shadow-[3px_3px_0px_0px_rgba(107,196,179,1)] group-hover:shadow-[4px_4px_0px_0px_rgba(107,196,179,1)] transition-all">
              <ShieldAlert className="h-6 w-6" />
            </div>
            <div className="flex flex-col items-start">
              <span className="text-2xl font-serif font-bold tracking-tight text-foreground leading-none">
                IR-AIS
              </span>
              <span className="text-[10px] font-sans font-bold uppercase tracking-wider text-muted-foreground leading-none mt-1 hidden sm:block">
                Intelligence System
              </span>
            </div>
          </button>

          {/* Desktop Nav Links */}
          <nav className="hidden md:flex items-center gap-3">
            {navLinks.map((link) => {
              const isActive = link.hash === currentTab;
              return (
                <button
                  key={link.hash}
                  onClick={() => handleNav(link)}
                  className={cn(
                    "tab-pill flex items-center gap-2 font-sans",
                    isActive ? "tab-pill-active" : "tab-pill-inactive"
                  )}
                >
                  <link.icon className="h-4 w-4" />
                  {link.label}
                  {isActive && <span className="ml-1">→</span>}
                </button>
              );
            })}
          </nav>

          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden border-2 border-black rounded-full h-10 w-10 bg-secondary/50"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle navigation menu"
          >
            {mobileOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </Button>
        </div>
      </header>

      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm md:hidden animate-fade-in"
          onClick={() => setMobileOpen(false)} 
        />
      )}

      {/* Mobile Menu */}
      <div className={cn(
        "fixed top-20 left-4 right-4 z-40 md:hidden transition-all duration-300 transform",
        mobileOpen ? "translate-y-0 opacity-100" : "-translate-y-4 opacity-0 pointer-events-none"
      )}>
        <div className="bg-card border-2 border-black rounded-2xl p-4 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
          <nav className="flex flex-col gap-2">
            {navLinks.map((link, i) => {
              const isActive = link.hash === currentTab;
              return (
                <button
                  key={link.hash}
                  onClick={() => handleNav(link)}
                  className={cn(
                    "flex items-center gap-3 px-4 py-4 rounded-xl text-sm font-bold border-2 transition-all duration-200",
                    isActive
                      ? "bg-primary border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] text-black"
                      : "bg-background border-transparent text-muted-foreground hover:border-black/20"
                  )}
                  style={{ animationDelay: `${i * 50}ms` }}
                >
                  <div className={cn(
                    "flex h-9 w-9 items-center justify-center rounded-full border-2 border-black transition-colors",
                    isActive ? "bg-white" : "bg-secondary"
                  )}>
                    <link.icon className="h-5 w-5" />
                  </div>
                  {link.label}
                  {isActive && (
                    <span className="ml-auto text-lg leading-none">→</span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </div>
    </>
  );
}
