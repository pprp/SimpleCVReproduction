#!/usr/bin/env python3
"""
Dataset file visualizer for terminal with navigation and search functionality.
Supports Parquet and JSONL formats commonly used in Hugging Face datasets.
"""

import argparse
import os
import re
import sys
from typing import Dict, Any, List
import pandas as pd
from colorama import Fore, Back, Style, init
import pyarrow.parquet as pq

# 提前初始化 colorama
init()


class DatasetVisualizer:
    """Terminal-based dataset file visualizer with navigation and search."""

    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        self.filepath = filepath
        
        # 尝试读取文件，支持多种格式
        file_ext = os.path.splitext(filepath)[1].lower()
        try:
            if file_ext == '.parquet':
                self.table = pq.read_table(filepath)
                self.df = self.table.to_pandas()
            elif file_ext == '.jsonl':
                # 支持 JSONL 格式
                import json
                data = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                self.df = pd.DataFrame(data)
            else:
                # 尝试作为 parquet 读取
                self.table = pq.read_table(filepath)
                self.df = self.table.to_pandas()
        except Exception as e:
            raise ValueError(f"Unable to read file {filepath}: {e}")
            
        if self.df.empty:
            raise ValueError(f"File {filepath} contains no data")
            
        self.current_row = 0
        self.rows_per_page = 10
        self.search_pattern = None
        self.search_results = []
        self.search_idx = -1

    # -------------------- 显示相关 --------------------
    def colorize_value(self, value: Any) -> str:
        if isinstance(value, str):
            # 截断过长的字符串
            display_str = value if len(value) <= 200 else value[:200] + "..."
            return f"{Fore.GREEN}{display_str}{Style.RESET_ALL}"
        elif isinstance(value, (int, float)):
            return f"{Fore.CYAN}{value}{Style.RESET_ALL}"
        elif isinstance(value, bool):
            return f"{Fore.YELLOW}{value}{Style.RESET_ALL}"
        elif isinstance(value, dict):
            return self.colorize_dict(value)
        elif isinstance(value, list):
            if len(value) > 5:
                preview = str(value[:5])[:-1] + f", ... and {len(value) - 5} more items]"
                return f"{Fore.MAGENTA}{preview}{Style.RESET_ALL}"
            else:
                return f"{Fore.MAGENTA}{value}{Style.RESET_ALL}"
        elif value is None:
            return f"{Fore.RED}None{Style.RESET_ALL}"
        else:
            return str(value)

    def colorize_dict(self, d: Dict[str, Any], indent: int = 0) -> str:
        if indent > 6:  # 防止过深的嵌套
            return f"{Fore.MAGENTA}...{Style.RESET_ALL}"
            
        result = "{\n"
        for k, v in d.items():
            spaces = " " * (indent + 2)
            key_str = f"{Fore.BLUE}{k}{Style.RESET_ALL}"

            if isinstance(v, dict):
                val_str = self.colorize_dict(v, indent + 2)
            elif isinstance(v, list) and v:
                if len(v) > 3:  # 限制显示的列表元素数量
                    sample_items = v[:3]
                    val_str = "[\n"
                    for item in sample_items:
                        if isinstance(item, dict):
                            val_str += f"{spaces}  {self.colorize_dict(item, indent + 4)},\n"
                        else:
                            val_str += f"{spaces}  {self.colorize_value(item)},\n"
                    val_str += f"{spaces}  {Fore.MAGENTA}... and {len(v) - 3} more items{Style.RESET_ALL}\n"
                    val_str += f"{spaces}]"
                else:
                    val_str = "[\n"
                    for item in v:
                        if isinstance(item, dict):
                            val_str += f"{spaces}  {self.colorize_dict(item, indent + 4)},\n"
                        else:
                            val_str += f"{spaces}  {self.colorize_value(item)},\n"
                    val_str += f"{spaces}]"
            else:
                val_str = self.colorize_value(v)

            result += f"{spaces}{key_str}: {val_str},\n"
        result += " " * indent + "}"
        return result

    def display_current_page(self) -> None:
        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"{Back.BLUE}{Fore.WHITE} Dataset Viewer: {os.path.basename(self.filepath)} {Style.RESET_ALL}")
        print(f"Rows: {len(self.df)} | Columns: {len(self.df.columns)}")
        
        # 计算当前页和总页数
        current_page = self.current_row // self.rows_per_page + 1
        total_pages = (len(self.df) - 1) // self.rows_per_page + 1
        print(f"Page: {current_page} of {total_pages} | Current row: {self.current_row}")

        if self.search_pattern:
            print(f"Search: '{self.search_pattern}' [{len(self.search_results)} matches]")
            if self.search_idx != -1 and len(self.search_results) > 0:
                print(f"Showing result {self.search_idx + 1} of {len(self.search_results)}")

        print("=" * 80)

        end_row = min(self.current_row + self.rows_per_page, len(self.df))
        for i in range(self.current_row, end_row):
            # 高亮搜索结果
            if self.search_idx != -1 and len(self.search_results) > 0 and i == self.search_results[self.search_idx]:
                print(f"{Back.YELLOW}{Fore.BLACK} Row {i} (SEARCH RESULT) {Style.RESET_ALL}")
            else:
                print(f"{Back.WHITE}{Fore.BLACK} Row {i} {Style.RESET_ALL}")

            row_data = self.df.iloc[i].to_dict()
            print(self.colorize_dict(row_data))
            print("-" * 80)

        print("\nCommands:")
        print("j/↓: next page | k/↑: prev page | g [num]: go to row")
        print("/[text]: search | //[text]: show all matches | n: next search | q: quit")

    # -------------------- 搜索 --------------------
    def search(self, pattern: str) -> None:
        """搜索包含指定模式的行"""
        self.search_pattern = pattern
        self.search_results = []
        self.search_idx = -1
        
        print(f"Searching for '{pattern}'...")
        for i, row in self.df.iterrows():
            # 将整行转换为字符串进行搜索
            row_str = str(row.to_dict()).lower()
            if pattern.lower() in row_str:
                self.search_results.append(i)
        
        # 显示搜索结果统计
        total_results = len(self.search_results)
        total_rows = len(self.df)
        
        if self.search_results:
            self.search_idx = 0
            current_result_row = self.search_results[0]
            # 跳转到包含第一个搜索结果的页面
            self.current_row = (current_result_row // self.rows_per_page) * self.rows_per_page
            
            print(f"✓ Found {total_results} matches out of {total_rows} total rows")
            print(f"  Showing result 1 of {total_results} (row {current_result_row})")
            print(f"  Use 'n' to navigate to next result")
        else:
            print(f"✗ No matches found for '{pattern}' in {total_rows} rows")

    def search_and_display_all(self, pattern: str) -> None:
        """搜索并显示所有匹配的行"""
        print(f"Searching for '{pattern}' and displaying all matches...")
        matches = []
        
        for i, row in self.df.iterrows():
            row_str = str(row.to_dict()).lower()
            if pattern.lower() in row_str:
                matches.append((i, row))
        
        total_results = len(matches)
        total_rows = len(self.df)
        
        if matches:
            print(f"\n{'='*80}")
            print(f"SEARCH RESULTS: Found {total_results} matches for '{pattern}' out of {total_rows} total rows")
            print(f"{'='*80}")
            
            for idx, (row_num, row_data) in enumerate(matches, 1):
                print(f"\n{Back.CYAN}{Fore.WHITE} Match {idx}/{total_results} - Row {row_num} {Style.RESET_ALL}")
                print(self.colorize_dict(row_data.to_dict()))
                print("-" * 80)
                
                # 每显示5个结果后询问是否继续
                if idx % 5 == 0 and idx < total_results:
                    print(f"\nShowing {idx} of {total_results} results...")
                    continue_display = input("Continue displaying results? (y/n, default=y): ").strip().lower()
                    if continue_display in {'n', 'no'}:
                        print(f"Stopped at {idx} of {total_results} results.")
                        break
                    
            print(f"\n{'='*80}")
            print(f"END OF SEARCH RESULTS ({total_results} matches)")
            print(f"{'='*80}")
        else:
            print(f"✗ No matches found for '{pattern}' in {total_rows} rows")

    def next_search_result(self) -> None:
        """跳转到下一个搜索结果"""
        if self.search_results and len(self.search_results) > 0:
            self.search_idx = (self.search_idx + 1) % len(self.search_results)
            current_result_row = self.search_results[self.search_idx]
            # 跳转到包含当前搜索结果的页面
            self.current_row = (current_result_row // self.rows_per_page) * self.rows_per_page
            
            total_results = len(self.search_results)
            print(f"→ Showing result {self.search_idx + 1} of {total_results} (row {current_result_row})")
            if self.search_idx == 0:
                print("  (Wrapped to first result)")
        else:
            print("✗ No active search results to navigate")

    # -------------------- 运行入口 --------------------
    def run(self) -> None:
        """运行数据集查看器，优先使用简单模式以确保兼容性"""
        # 直接使用简单模式，更稳定
        self._run_simple()

    # -------------------- 简单交互 --------------------
    def _run_simple(self) -> None:
        while True:
            self.display_current_page()
            try:
                cmd = input("\nEnter command: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if cmd.lower() in {'q', 'quit', 'exit'}:
                break
            elif cmd.lower() in {'j', 'down', ''}:  # 空回车也表示下一页
                if self.current_row + self.rows_per_page < len(self.df):
                    self.current_row += self.rows_per_page
                else:
                    print("Already at the last page. Press any key to continue...")
                    input()
            elif cmd.lower() in {'k', 'up'}:
                if self.current_row >= self.rows_per_page:
                    self.current_row -= self.rows_per_page
                else:
                    print("Already at the first page. Press any key to continue...")
                    input()
            elif cmd.lower().startswith('g '):
                try:
                    row_num = int(cmd.split()[1])
                    if 0 <= row_num < len(self.df):
                        # 计算包含该行的页面起始位置
                        self.current_row = (row_num // self.rows_per_page) * self.rows_per_page
                    else:
                        print(f"Row number must be between 0 and {len(self.df) - 1}. Press any key to continue...")
                        input()
                except (ValueError, IndexError):
                    print("Invalid row number. Usage: g [row_number]. Press any key to continue...")
                    input()
            elif cmd.startswith('//'):
                # 新功能：显示所有搜索结果
                search_term = cmd[2:].strip()
                if search_term:
                    self.search_and_display_all(search_term)
                    print("\nPress any key to continue...")
                    input()
                else:
                    print("Please enter a search term. Usage: //[search_term]. Press any key to continue...")
                    input()
            elif cmd.startswith('/'):
                search_term = cmd[1:].strip()
                if search_term:
                    self.search(search_term)
                    if not self.search_results:
                        print("Press any key to continue...")
                        input()
                    else:
                        print("Press any key to continue...")
                        input()
                else:
                    print("Please enter a search term. Usage: /[search_term]. Press any key to continue...")
                    input()
            elif cmd.lower() == 'n':
                if self.search_results:
                    self.next_search_result()
                    # 确保当前行在正确的页面上
                    current_result_row = self.search_results[self.search_idx]
                    self.current_row = (current_result_row // self.rows_per_page) * self.rows_per_page
                    print("Press any key to continue...")
                    input()
                else:
                    print("No active search results. Please search first using /[text]. Press any key to continue...")
                    input()
            elif cmd.lower() == 'h' or cmd.lower() == 'help':
                print("\nAvailable commands:")
                print("  j, down, [Enter] - Next page")
                print("  k, up - Previous page")
                print("  g [number] - Go to specific row")
                print("  /[text] - Search for text (navigate with 'n')")
                print("  //[text] - Search and display ALL matches at once")
                print("  n - Next search result (after using /[text])")
                print("  h, help - Show this help")
                print("  q, quit, exit - Quit the viewer")
                print("\nSearch Examples:")
                print("  /math - Find rows containing 'math'")
                print("  //triangle - Show all rows containing 'triangle'")
                print("\nPress any key to continue...")
                input()
            else:
                print("Unknown command. Type 'h' for help. Press any key to continue...")
                input()

    # -------------------- curses 交互 --------------------
    def _run_curses(self) -> None:
        import curses  # 确保只在需要时才 import

        def main(stdscr):
            curses.curs_set(0)
            stdscr.clear()
            curses.start_color()
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_YELLOW)

            while True:
                stdscr.clear()
                self.display_current_page()
                key = stdscr.getch()

                if key == ord('q'):
                    break
                elif key in (curses.KEY_DOWN, ord('j')):
                    if self.current_row + self.rows_per_page < len(self.df):
                        self.current_row += self.rows_per_page
                elif key in (curses.KEY_UP, ord('k')):
                    self.current_row = max(0, self.current_row - self.rows_per_page)
                elif key == ord('g'):
                    stdscr.addstr(curses.LINES - 1, 0, "Go to row: ")
                    curses.echo()
                    curses.curs_set(1)
                    row_str = stdscr.getstr().decode('utf-8')
                    curses.noecho()
                    curses.curs_set(0)
                    try:
                        row_num = int(row_str)
                        if 0 <= row_num < len(self.df):
                            self.current_row = row_num
                    except ValueError:
                        pass
                elif key == ord('/'):
                    stdscr.addstr(curses.LINES - 1, 0, "Search: ")
                    curses.echo()
                    curses.curs_set(1)
                    pattern = stdscr.getstr().decode('utf-8')
                    curses.noecho()
                    curses.curs_set(0)
                    self.search(pattern)
                elif key == ord('n'):
                    self.next_search_result()

        curses.wrapper(main)


# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser(description='Terminal dataset file visualizer')
    parser.add_argument('file', help='Path to the dataset file to visualize (supports .parquet and .jsonl)')
    args = parser.parse_args()

    try:
        visualizer = DatasetVisualizer(args.file)
        visualizer.run()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
