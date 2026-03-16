import { useTheme } from "next-themes";
import { Button } from "@/app/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/app/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/app/components/ui/avatar";
import {
  Home,
  LayoutDashboard,
  Wand2,
  Settings,
  LogOut,
  Sun,
  Moon,
} from "lucide-react";
import { cn } from "@/app/components/ui/utils";

interface HeaderProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  onNewConvert: () => void;
}

export function Header({ currentPage, onNavigate, onNewConvert }: HeaderProps) {
  const { theme, setTheme } = useTheme();

  const navItems = [
    { id: "home", label: "Home", icon: Home },
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "convert", label: "Convert", icon: Wand2 },
  ];

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60 shadow-sm">
      <div className="container mx-auto flex h-14 items-center justify-between px-4 lg:px-6">
        {/* Logo */}
        <div
          className="flex items-center gap-2 cursor-pointer"
          onClick={() => onNavigate("home")}
        >
          <div className="flex items-center gap-2 text-primary">
            <Wand2 className="h-5 w-5" />
            <span className="hidden sm:inline text-base font-semibold">
              MathSnap
            </span>
            <span className="sm:hidden text-base font-semibold">MS</span>
          </div>
        </div>

        {/* Desktop + Tablet Navigation */}
        <nav className="hidden md:flex items-center gap-1">
          {navItems.map((item) => (
            <Button
              key={item.id}
              variant={currentPage === item.id ? "default" : "ghost"}
              size="sm"
              onClick={() => onNavigate(item.id)}
              className={cn(
                "gap-1.5 h-8 text-xs",
                currentPage === item.id &&
                  "bg-primary text-primary-foreground"
              )}
            >
              <item.icon className="h-3.5 w-3.5" />
              <span className="hidden lg:inline">{item.label}</span>
            </Button>
          ))}
        </nav>

        {/* Right Side Actions */}
        <div className="flex items-center gap-1.5">
          <Button
            onClick={onNewConvert}
            size="sm"
            className="bg-primary text-primary-foreground hover:bg-primary/90 hidden sm:flex h-8 text-xs"
          >
            <Wand2 className="h-3.5 w-3.5 mr-1.5" />
            New Convert
          </Button>

          {/* Dark Mode Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="h-8 w-8"
          >
            <Sun className="h-4 w-4 rotate-0 scale-100 transition-transform dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-transform dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>

          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="rounded-full h-8 w-8">
                <Avatar className="h-7 w-7">
                  <AvatarFallback className="bg-primary text-primary-foreground text-xs">
                    JD
                  </AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem onClick={() => onNavigate("settings")}>
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <LogOut className="h-4 w-4 mr-2" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Mobile Bottom Navigation */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 bg-card border-t border-border z-50">
        <nav className="flex items-center justify-around px-2 py-1.5">
          {navItems.map((item) => (
            <Button
              key={item.id}
              variant="ghost"
              size="sm"
              onClick={() => onNavigate(item.id)}
              className={cn(
                "flex flex-col gap-0.5 h-auto py-1.5 px-3",
                currentPage === item.id && "text-primary"
              )}
            >
              <item.icon className="h-4 w-4" />
              <span className="text-[10px]">{item.label}</span>
            </Button>
          ))}
        </nav>
      </div>
    </header>
  );
}
