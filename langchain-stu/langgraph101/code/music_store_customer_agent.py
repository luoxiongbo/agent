import ast
import sqlite3
import sys
import uuid
from pathlib import Path

import requests
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.models import model


def get_engine_for_chinook_db():
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"

    response = requests.get(url)
    sql_script = response.text

    # 这里使用 :memory:，意味着数据库运行在内存中，不需要创建实际的数据库文件。
    connection = sqlite3.connect(
        ":memory:",
        check_same_thread=False
    )
    connection.executescript(sql_script)

    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

# 长期记忆
in_memory_store = InMemoryStore()

# 短期记忆
checkpointer = MemorySaver()


class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class State(InputState):
    customer_id: int
    loaded_memory: str
    remaining_steps: int


@tool
def get_albums_by_artist(artist: str):
    """Get albums by an artist."""
    return db.run(
        f"""
        SELECT Album.Title, Artist.Name
        FROM Album
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True
    )


@tool
def get_tracks_by_artist(artist: str):
    """Get songs by an artist (or similar artists)."""
    return db.run(
        f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName
        FROM Album
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True
    )


@tool
def get_songs_by_genre(genre: str):
    """
    Fetch songs from the database that match a specific genre.

    Args:
        genre (str): The genre of the songs to fetch.

    Returns:
        list[dict]: A list of songs that match the specified genre.
    """
    genre_id_query = f"SELECT GenreId FROM Genre WHERE Name LIKE '%{genre}%'"
    genre_ids = db.run(genre_id_query)
    if not genre_ids:
        return f"No songs found for the genre: {genre}"
    genre_ids = ast.literal_eval(genre_ids)
    genre_id_list = ", ".join(str(gid[0]) for gid in genre_ids)

    songs_query = f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName
        FROM Track
        LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.GenreId IN ({genre_id_list})
        GROUP BY Artist.Name
        LIMIT 8;
    """
    songs = db.run(songs_query, include_columns=True)
    if not songs:
        return f"No songs found for the genre: {genre}"
    formatted_songs = ast.literal_eval(songs)
    return [
        {"Song": song["SongName"], "Artist": song["ArtistName"]}
        for song in formatted_songs
    ]


@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return db.run(
        f"""
        SELECT * FROM Track WHERE Name LIKE '%{song_title}%';
        """,
        include_columns=True
    )


music_tools = [
    get_albums_by_artist,
    get_tracks_by_artist,
    get_songs_by_genre,
    check_for_songs
]

llm_with_music_tools = model.bind_tools(music_tools)

music_tool_node = ToolNode(music_tools)


def generate_music_assistant_prompt(memory: str = "None") -> str:
    return f"""
    You are a member of the assistant team, your role specifically is to focused on helping customers discover and learn about music in our digital catalog.
    If you are unable to find playlists, songs, or albums associated with an artist, it is okay.
    Just inform the customer that the catalog does not have any playlists, songs, or albums associated with that artist.
    You also have context on any saved user preferences, helping you to tailor your response.

    CORE RESPONSIBILITIES:
    - Search and provide accurate information about songs, albums, artists, and playlists
    - Offer relevant recommendations based on customer interests
    - Handle music-related queries with attention to detail
    - Help customers discover new music they might enjoy
    - You are routed only when there are questions related to music catalog; ignore other questions.

    SEARCH GUIDELINES:
    1. Always perform thorough searches before concluding something is unavailable
    2. If exact matches aren't found, try:
       - Checking for alternative spellings
       - Looking for similar artist names
       - Searching by partial matches
       - Checking different versions/remixes
    3. When providing song lists:
       - Include the artist name with each song
       - Mention the album when relevant
       - Note if it's part of any playlists
       - Indicate if there are multiple versions

    Additional context is provided below:

    Prior saved user preferences: {memory}

    Message history is also attached.
    """


def music_assistant(state: State):
    memory = "None"
    if "loaded_memory" in state:
        memory = state["loaded_memory"]

    music_assistant_prompt = generate_music_assistant_prompt(memory)

    response = llm_with_music_tools.invoke(
        [SystemMessage(music_assistant_prompt)]
        + state["messages"]
    )

    return {"messages": [response]}


def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


music_workflow = StateGraph(State)

music_workflow.add_node(
    "music_assistant",
    music_assistant
)

music_workflow.add_node(
    "music_tool_node",
    music_tool_node
)

music_workflow.add_edge(
    START,
    "music_assistant"
)

music_workflow.add_conditional_edges(
    "music_assistant",
    should_continue,
    {
        "continue": "music_tool_node",
        "end": END,
    },
)

music_workflow.add_edge(
    "music_tool_node",
    "music_assistant"
)

music_catalog_subagent = music_workflow.compile(
    name="music_catalog_subagent",
    checkpointer=checkpointer,
    store=in_memory_store
)

question = """
I like the Rolling Stones.
What songs do you recommend by them
or by other artists that I might like?
"""

config = {"configurable": {"thread_id": uuid.uuid4()}}

result = music_catalog_subagent.invoke(
    {"messages": [HumanMessage(content=question)]},
    config=config
)

for message in result["messages"]:
    message.pretty_print()
