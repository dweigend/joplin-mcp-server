"""CLI für die Joplin API mit interaktivem Menü."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt, IntPrompt
from rich.table import Table

from joplin_api import JoplinAPI, JoplinNote, OrderDirection
from joplin_utils import get_token_from_env, read_markdown_file

# Rich Console Setup
console = Console()
app = typer.Typer()

MENU_OPTIONS = {
    "1": ("📝 Notizen auflisten", "list_notes"),
    "2": ("✨ Neue Notiz erstellen", "create_note"),
    "3": ("👀 Notiz anzeigen", "view_note"),
    "4": ("✏️ Notiz bearbeiten", "update_note"),
    "5": ("🗑️ Notiz löschen", "delete_note"),
    "6": ("🔍 Notizen durchsuchen", "search_notes"),
    "0": ("👋 Beenden", "exit")
}

def show_menu() -> str:
    """Zeigt das Hauptmenü an und gibt die Auswahl zurück."""
    console.print("\n[bold blue]🗒️ Joplin Notizen Manager[/]\n")
    
    for key, (label, _) in MENU_OPTIONS.items():
        console.print(f"[bold cyan]{key}[/] - {label}")
    
    choice = Prompt.ask(
        "\n[bold green]Wähle eine Option[/]",
        choices=list(MENU_OPTIONS.keys())
    )
    return MENU_OPTIONS[choice][1]

def get_api() -> JoplinAPI:
    """Initialize and return the Joplin API client.

    Returns:
        JoplinAPI instance

    Raises:
        typer.Exit: If initialization fails
    """
    try:
        token = get_token_from_env()
        return JoplinAPI(token=token)
    except Exception as e:
        console.print(f"[red]❌ Fehler beim Initialisieren der API: {e}[/]")
        raise typer.Exit(1)

def create_notes_table(notes: list[JoplinNote]) -> Table:
    """Erstelle eine formatierte Tabelle für Notizen."""
    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("📌 ID", style="dim")
    table.add_column("📝 Titel", style="bold")
    table.add_column("📅 Erstellt", justify="right")
    table.add_column("🔄 Aktualisiert", justify="right")
    table.add_column("✅ Todo", justify="center")

    for note in notes:
        table.add_row(
            note.id,
            note.title,
            format_date(note.created_time),
            format_date(note.updated_time),
            "✓" if note.is_todo else ""
        )

    return table

def format_date(dt: datetime | None) -> str:
    """Formatiere ein Datum für die Anzeige.
    
    Args:
        dt: Das zu formatierende Datum oder None
        
    Returns:
        Formatiertes Datum oder 'N/A' wenn None
    """
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "N/A"

def list_notes():
    """Liste alle Notizen auf."""
    api = get_api()

    page = IntPrompt.ask("📄 Welche Seite möchtest du sehen?", default=1)
    limit = IntPrompt.ask("📊 Wie viele Notizen pro Seite?", default=10)
    ascending = Confirm.ask("⬆️ Aufsteigend sortieren?", default=False)

    try:
        with console.status("[bold green]📂 Lade Notizen..."):
            response = api.get_notes(
                page=page,
                limit=limit,
                order_by="updated_time",
                order_dir=OrderDirection.ASC if ascending else OrderDirection.DESC
            )

        if not response.items:
            console.print("[yellow]📭 Keine Notizen gefunden.[/]")
            return

        table = create_notes_table(response.items)
        console.print(table)

        if response.has_more:
            console.print("\n[blue]📚 Es gibt weitere Notizen. Nutze eine höhere Seitenzahl für mehr.[/]")

    except Exception as e:
        console.print(f"[red]❌ Fehler beim Laden der Notizen: {e}[/]")

def create_note():
    """Erstelle eine neue Notiz."""
    api = get_api()

    try:
        title = Prompt.ask("📝 Titel der Notiz")
        is_todo = Confirm.ask("✅ Als Todo markieren?", default=False)

        console.print("\n📝 Inhalt (STRG+D zum Beenden):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            body = "\n".join(lines)

        with console.status("[bold green]✨ Erstelle Notiz..."):
            note = api.create_note(title=title, body=body, is_todo=is_todo)

        console.print("\n[green]✅ Notiz erfolgreich erstellt:[/]")
        console.print(Panel(
            f"[bold]📝 Titel:[/] {note.title}\n"
            f"[bold]🔑 ID:[/] {note.id}\n"
            f"[bold]📅 Erstellt:[/] {format_date(note.created_time)}\n"
            f"[bold]✅ Todo:[/] {'Ja' if note.is_todo else 'Nein'}",
            title="✨ Neue Notiz",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"[red]❌ Fehler beim Erstellen der Notiz: {e}[/]")

def view_note():
    """Zeige eine spezifische Notiz an."""
    api = get_api()

    note_id = Prompt.ask("🔑 Gib die ID der Notiz ein")
    raw = Confirm.ask("📄 Markdown-Inhalt ohne Formatierung anzeigen?", default=False)

    try:
        with console.status("[bold green]📂 Lade Notiz..."):
            note = api.get_note(note_id)
            
        # Debug-Informationen
        console.print("\n[dim]🔍 Debug: API-Antwort:[/]")
        console.print(f"[dim]- Type von body: {type(note.body)}[/]")
        console.print(f"[dim]- Länge von body: {len(note.body) if note.body else 0}[/]")
        if note.body:
            console.print(f"[dim]- Erste 50 Zeichen: {note.body[:50]}...[/]")

        console.print(Panel(
            f"[bold]📝 Titel:[/] {note.title}\n"
            f"[bold]🔑 ID:[/] {note.id}\n"
            f"[bold]📅 Erstellt:[/] {format_date(note.created_time)}\n"
            f"[bold]🔄 Aktualisiert:[/] {format_date(note.updated_time)}\n"
            f"[bold]✅ Todo:[/] {'Ja' if note.is_todo else 'Nein'}",
            title="📋 Notizdetails",
            border_style="blue"
        ))

        if note.body:
            if raw:
                console.print("\n[bold]📄 Inhalt:[/]")
                console.print(note.body)
            else:
                console.print("\n[bold]📝 Formatierter Inhalt:[/]")
                console.print(Markdown(note.body))
        else:
            console.print("\n[yellow]📭 Diese Notiz hat keinen Inhalt.[/]")

    except Exception as e:
        console.print(f"[red]❌ Fehler beim Laden der Notiz: {e}[/]")
        # Bei Fehlern mehr Details anzeigen
        console.print(f"[dim]🔍 Debug: Exception Details:[/]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/]")

def update_note():
    """Aktualisiere eine existierende Notiz."""
    api = get_api()

    note_id = Prompt.ask("🔑 Gib die ID der Notiz ein")

    try:
        with console.status("[bold green]📂 Lade Notiz..."):
            note = api.get_note(note_id)

        console.print(f"\n[bold]📝 Aktuelle Notiz:[/] {note.title}")

        title = Prompt.ask(
            "📝 Neuer Titel",
            default=note.title,
            show_default=True
        )

        console.print("\n📝 Neuer Inhalt (STRG+D zum Beenden):")
        console.print(f"[dim]Aktuell: {len(note.body or '') } Zeichen[/]")

        if note.body:
            console.print(note.body)

        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            body = "\n".join(lines) if lines else note.body

        is_todo = Confirm.ask(
            "✅ Als Todo markieren?",
            default=note.is_todo
        )

        with console.status("[bold green]🔄 Aktualisiere Notiz..."):
            note = api.update_note(
                note_id=note_id,
                title=title,
                body=body,
                is_todo=is_todo
            )

        console.print("\n[green]✅ Notiz erfolgreich aktualisiert![/]")

    except Exception as e:
        console.print(f"[red]❌ Fehler beim Aktualisieren der Notiz: {e}[/]")

def delete_note():
    """Lösche eine Notiz."""
    api = get_api()

    note_id = Prompt.ask("🔑 Gib die ID der Notiz ein")
    permanent = Confirm.ask("🗑️ Permanent löschen (nicht in Papierkorb)?", default=False)

    try:
        with console.status("[bold green]📂 Lade Notiz..."):
            note = api.get_note(note_id)

        console.print(Panel(
            f"[bold]📝 Titel:[/] {note.title}\n"
            f"[bold]🔑 ID:[/] {note.id}",
            title="🗑️ Zu löschende Notiz",
            border_style="red"
        ))

        if not Confirm.ask(
            "[red]❗ Möchtest du diese Notiz wirklich löschen?[/]",
            default=False
        ):
            console.print("[yellow]↩️ Löschen abgebrochen.[/]")
            return

        with console.status(
            "[bold red]🗑️ Lösche Notiz..."
            if permanent
            else "[bold yellow]🗑️ Verschiebe Notiz in Papierkorb..."
        ):
            api.delete_note(note_id, permanent=permanent)

        console.print("\n[green]✅ Notiz erfolgreich gelöscht![/]")

    except Exception as e:
        console.print(f"[red]❌ Fehler beim Löschen der Notiz: {e}[/]")

def search_notes():
    """Durchsuche alle Notizen."""
    api = get_api()

    query = Prompt.ask("🔍 Wonach möchtest du suchen")
    limit = IntPrompt.ask("📊 Maximale Anzahl von Ergebnissen", default=10)

    try:
        with console.status(f'[bold green]🔍 Suche nach "{query}"...'):
            response = api.search_notes(query=query, limit=limit)

        if not response.items:
            console.print("[yellow]📭 Keine Notizen gefunden.[/]")
            return

        table = create_notes_table(response.items)
        console.print(table)

        if response.has_more:
            console.print("\n[blue]📚 Es gibt weitere Ergebnisse. Erhöhe die maximale Anzahl für mehr.[/]")

    except Exception as e:
        console.print(f"[red]❌ Fehler bei der Suche: {e}[/]")

def main():
    """Hauptprogramm mit Menüführung."""
    console.print("[bold green]🚀 Willkommen beim Joplin Notizen Manager![/]")
    
    while True:
        try:
            choice = show_menu()
            
            if choice == "exit":
                console.print("[bold blue]👋 Auf Wiedersehen![/]")
                break
                
            # Funktion aus dem Menü aufrufen
            globals()[choice]()
            
            # Warte auf Bestätigung bevor das Menü wieder angezeigt wird
            Prompt.ask("\n[bold cyan]↩️ Drücke Enter um fortzufahren[/]")
            
        except Exception as e:
            console.print(f"[red]❌ Ein unerwarteter Fehler ist aufgetreten: {e}[/]")
            if Confirm.ask("[yellow]🔄 Möchtest du es erneut versuchen?[/]", default=True):
                continue
            break

if __name__ == "__main__":
    main()
