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
  LogOut
} from "lucide-react"; // Đã xóa các icon không dùng để nhẹ code
import { cn } from "@/app/components/ui/utils";

interface HeaderProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  onNewConvert: () => void;
}

export function Header({ currentPage, onNavigate, onNewConvert }: HeaderProps) {
  // Chỉ giữ lại 3 mục bà muốn
  const navItems = [
    { id: "home", label: "Home", icon: Home },
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "convert", label: "Convert", icon: Wand2 },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-card shadow-sm">
      <div className="container mx-auto flex h-16 items-center justify-between px-4 lg:px-6">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 text-primary">
            <Wand2 className="h-6 w-6" />
            <span className="hidden sm:inline text-lg font-semibold">Math → LaTeX</span>
            <span className="sm:hidden text-lg font-semibold">M→L</span>
          </div>
        </div>

        {/* Desktop Navigation */}
        <nav className="hidden lg:flex items-center gap-1">
          {navItems.map((item) => (
            <Button
              key={item.id}
              variant={currentPage === item.id ? "default" : "ghost"}
              size="sm"
              onClick={() => onNavigate(item.id)}
              className={cn(
                "gap-2",
                currentPage === item.id && "bg-primary text-primary-foreground"
              )}
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </Button>
          ))}
        </nav>

        {/* Right Side Actions */}
        <div className="flex items-center gap-2">
          <Button 
            onClick={onNewConvert}
            size="sm"
            className="bg-primary text-primary-foreground hover:bg-primary/90 hidden sm:flex"
          >
            <Wand2 className="h-4 w-4 mr-2" />
            New Convert
          </Button>

          {/* User Menu - Đã dọn dẹp các mục Projects/History dư thừa */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="rounded-full">
                <Avatar className="h-8 w-8">
                  <AvatarFallback className="bg-primary text-primary-foreground">JD</AvatarFallback>
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

      {/* Mobile Bottom Navigation - Tự động hiển thị theo navItems */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 bg-card border-t border-border z-50">
        <nav className="flex items-center justify-around px-2 py-2">
          {navItems.map((item) => (
            <Button
              key={item.id}
              variant="ghost"
              size="sm"
              onClick={() => onNavigate(item.id)}
              className={cn(
                "flex flex-col gap-1 h-auto py-2 px-3",
                currentPage === item.id && "text-primary"
              )}
            >
              <item.icon className="h-5 w-5" />
              <span className="text-xs">{item.label}</span>
            </Button>
          ))}
        </nav>
      </div>
    </header>
  );
}