# Scrapr

A Python-based data scraper that can extract content from PDF, Word (.docx), and plain text (.txt) files, and save the extracted content into a Markdown (.md) file.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/acwalkr/Scrapr.git
    cd Scrapr
    ```
2.  (Optional, but recommended) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use Scrapr, run the `scraper.py` script from your terminal.

You can provide the path to the file you want to scrape as a command-line argument:

```bash
python scraper.py path/to/your/document.pdf
```
Or for other file types:
```bash
python scraper.py path/to/your/document.docx
```
```bash
python scraper.py path/to/your/document.txt
```

If you run the script without providing a file path, it will prompt you to enter one:
```bash
python scraper.py
```
It will then ask: `Please enter the path to the file: `

The extracted content will be saved as a Markdown (`.md`) file in the same directory as the original document. The new file will have the same base name as the original file (e.g., `document.pdf` will result in `document.md`).

## Key Binding Integration (Conceptual Outline)

This project includes a `run_scraper.py` script, which serves as a basic example of how you might call the main scraper. The goal is to integrate this scraper with a system-wide key binding (hotkey) for quick activation. Since modifying system settings is beyond the scope of this tool, this section provides a conceptual guide.

The core idea is to have a hotkey trigger a small script (OS-specific) that:
1.  Determines the file path of the currently active/selected document.
2.  Executes the Python scraper, passing this file path as an argument.

Example command to be triggered by the hotkey:
```bash
python /path/to/Scrapr/scraper.py <path_to_active_document>
```
Or, if using the example runner:
```bash
python /path/to/Scrapr/run_scraper.py <path_to_active_document>
```
(Note: `run_scraper.py` would need to be modified to accept and use this argument instead of its current hypothetical path).

### Operating System Specific Guidance:

*   **Windows (using AutoHotkey)**:
    1.  Install [AutoHotkey](https://www.autohotkey.com/).
    2.  Create an AutoHotkey script (e.g., `scrape_hotkey.ahk`):
        ```ahk
        ; Example: Ctrl+Alt+S hotkey
        ^!s::
        ; Attempt to get the path of the selected file in Explorer
        SelectedFile := ""
        WinGet, ActivePID, PID, A
        for process in ComObjGet("winmgmts:").ExecQuery("SELECT * FROM Win32_Process WHERE ProcessId = " . ActivePID)
        {
            ParentPID := process.ParentProcessId
            for processParent in ComObjGet("winmgmts:").ExecQuery("SELECT * FROM Win32_Process WHERE ProcessId = " . ParentPID)
            {
                if (processParent.Name = "explorer.exe")
                {
                    ControlGetFocus, FocusedControl, A
                    if InStr(FocusedControl, "DirectUIHWND") or InStr(FocusedControl, "SysListView32")
                    {
                        for item in ComObjCreate("Shell.Application").Windows()
                        {
                            if (item.hwnd == WinExist("A"))
                            {
                                SelectedFile := item.Document.FocusedItem.Path
                                break
                            }
                        }
                    }
                    break
                }
            }
            if (SelectedFile != "")
                break
        }

        if (SelectedFile = "")
        {
            MsgBox, Could not determine the selected file. Please select a file in Explorer.
            return
        }
        
        ; Path to your Python executable (if not in PATH) and scraper.py
        PythonExe := "python.exe" ; or full path, e.g., "C:\Python39\python.exe"
        ScraperScript := "C:\path\to\Scrapr\scraper.py" ; Change this to your actual path

        Run, %PythonExe% "%ScraperScript%" "%SelectedFile%"
        return
        ```
    3.  Run this `.ahk` script. Now, pressing `Ctrl+Alt+S` (or your chosen hotkey) will attempt to run the scraper on the selected file in File Explorer.
    *Note: Getting the active file path reliably across all applications is complex. The example above is primarily for File Explorer. More sophisticated solutions might involve UI automation libraries.*

*   **macOS (using Automator and AppleScript)**:
    1.  Open Automator and create a new "Quick Action".
    2.  Set "Workflow receives current" to "files or folders" in "Finder.app".
    3.  Add a "Run Shell Script" action.
    4.  Set "Pass input" to "as arguments".
    5.  Use the following script:
        ```bash
        # The first argument ($1) will be the path to the selected file
        /usr/local/bin/python3 /path/to/Scrapr/scraper.py "$1" 
        # Make sure python3 points to your Python 3 installation
        # and update /path/to/Scrapr/scraper.py to the actual script path
        ```
    6.  Save the Quick Action (e.g., "Run Scrapr").
    7.  Go to System Preferences > Keyboard > Shortcuts > Services. Find your Quick Action and assign a keyboard shortcut.

*   **Linux (using custom desktop entry or tools like `xbindkeys`)**:
    *   **Desktop Entry (for file managers that support it):**
        Many file managers allow you to add custom actions to context menus. This isn't a global hotkey but provides quick access.
    *   **Using `xbindkeys` and a custom script:**
        1.  Install `xbindkeys` (e.g., `sudo apt-get install xbindkeys`).
        2.  Create a script (e.g., `~/.scrapr_launcher.sh`):
            ```bash
            #!/bin/bash
            # This is tricky, as getting the "currently selected file" globally is not straightforward.
            # One common approach is to use `xclip` or `xsel` to get the path if it's been copied to the clipboard.
            # Or, if you use a specific file manager, it might have a command-line way to get the selection.
            # For a more robust solution, you might need a small utility that integrates with your window manager
            # or desktop environment to get the active file path.

            # Simplistic example: assumes the file path is in the clipboard
            # FILE_PATH=$(xclip -o -selection clipboard) 
            # A better way would be needed for a real application.
            # For now, let's assume you'll manually provide it or use a file dialog if not passed.
            
            # This example will just call scraper.py, which will prompt if no arg is given.
            # To make it truly useful, you'd need a reliable way to get the active file.
            # For instance, if you could get it via a command like `getActiveFilePathCmd`:
            # ACTIVE_FILE_PATH=$(getActiveFilePathCmd)
            # if [ -n "$ACTIVE_FILE_PATH" ]; then
            #    python /path/to/Scrapr/scraper.py "$ACTIVE_FILE_PATH"
            # else
            #    # Fallback: Open terminal and let scraper.py prompt for input
            #    gnome-terminal -- python /path/to/Scrapr/scraper.py 
            # fi

            # For demonstration, this will just run scraper.py which will prompt.
            # You'd replace this with a mechanism to get the file path.
            python /path/to/Scrapr/scraper.py 
            ```
            Make it executable: `chmod +x ~/.scrapr_launcher.sh`.
        3.  Configure `xbindkeys`. Create or edit `~/.xbindkeysrc`:
            ```
            # Scrapr hotkey (e.g., Ctrl+Alt+S)
            "/home/youruser/.scrapr_launcher.sh"
               m:0x1c + c:39 
               Control+Alt + s
            ```
            *(You can get keycodes using `xbindkeys -k`)*
        4.  Run `xbindkeys`.

This section provides a starting point. The exact method for obtaining the active file path can be complex and may require additional tools or scripting specific to your desktop environment or primary applications.

## Running Tests

To run the unit tests:

1.  Navigate to the project's root directory.
2.  Run the test script:
    ```bash
    python tests/test_scraper.py
    ```
