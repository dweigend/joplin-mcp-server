# 📝 Joplin MCP Server

A Model Context Protocol (MCP) Server for [Joplin](https://joplinapp.org/) that enables note access through the [Model Context Protocol](https://modelcontextprotocol.io). Perfect for integration with AI assistants like Claude.

## ✨ Features

- 🔍 **Search Notes**: Full-text search across all notes
- 📖 **Read Notes**: Retrieve individual notes
- ✏️ **Edit Notes**: Create new notes and update existing ones
- 🗑️ **Delete Notes**: Move notes to trash or delete permanently
- 📥 **Markdown Import**: Import markdown files as notes
- 🤖 **AI Integration**: Seamless integration with Claude and other MCP-capable AI assistants

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- [Joplin Desktop](https://joplinapp.org/) with Web Clipper Service enabled
- [uv](https://github.com/astral-sh/uv) (Python package manager)

```bash
# Clone repository
git clone https://github.com/dweigend/joplin-mcp.git
cd joplin-mcp

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
```bash
uv pip install -e .
```

## ⚙️ Configuration

### Joplin API Token

1. Open Joplin Desktop
2. Go to Tools -> Options -> Web Clipper
3. Enable the Web Clipper Service
4. Copy the API Token

Create a `.env` file in the project directory:
```bash
JOPLIN_TOKEN=your_api_token_here
```

### Claude Desktop Setup

1. **Install Claude Desktop**
   - Download [Claude Desktop](https://claude.ai/download)
   - Ensure you have the latest version (Menu: Claude -> Check for Updates...)

2. **Configure MCP Server**
   ```json
   {
     "mcpServers": {
       "joplin": {
         "command": "/PATH/TO/UV/uv",
         "args": [
           "--directory",
           "/PATH/TO/YOUR/PROJECT/joplin_mcp",
           "run",
           "src/mcp/joplin_mcp.py"
         ]
       }
     }
   }
   ```
   - Replace `/PATH/TO/UV/uv` with the absolute path to your uv installation
     - Find the path with: `which uv`
     - Example macOS: `/Users/username/.local/bin/uv`
     - Example Windows: `C:\Users\username\AppData\Local\Microsoft\WindowsApps\uv.exe`
   - Replace `/PATH/TO/YOUR/PROJECT/joplin_mcp` with the absolute path to your project

   **Important**: Claude Desktop needs the full path to `uv` as it cannot access shell environment variables.

## 🛠️ Available Tools

### search_notes
Search for notes in Joplin.

**Parameters:**
- `query` (string): Search query
- `limit` (int, optional): Maximum number of results (default: 100)

### get_note
Retrieve a specific note by its ID.

**Parameters:**
- `note_id` (string): ID of the note

### create_note
Create a new note.

**Parameters:**
- `title` (string): Note title
- `body` (string, optional): Note content in Markdown
- `parent_id` (string, optional): ID of parent folder
- `is_todo` (boolean, optional): Whether this is a todo item

### update_note
Update an existing note.

**Parameters:**
- `note_id` (string): ID of note to update
- `title` (string, optional): New title
- `body` (string, optional): New content
- `parent_id` (string, optional): New parent folder ID
- `is_todo` (boolean, optional): New todo status

### delete_note
Delete a note.

**Parameters:**
- `note_id` (string): ID of note to delete
- `permanent` (boolean, optional): If true, permanently delete the note

### import_markdown
Import a markdown file as a new note.

**Parameters:**
- `file_path` (string): Path to the markdown file

## 🧪 Development

### Debug Mode

To start the server in debug mode:

```bash
MCP_LOG_LEVEL=debug mcp dev src/mcp/joplin_mcp.py
```

This starts the MCP Inspector at http://localhost:5173 where you can test the tools.

## 📄 License

[MIT License](LICENSE) - Copyright (c) 2025 David Weigend

## 👤 Author

**David Weigend**

* Website: [weigend.studio](https://weigend.studio)
* GitHub: [@dweigend](https://github.com/dweigend)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!
Visit the [issues page](https://github.com/dweigend/joplin-mcp/issues).