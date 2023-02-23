# taken from https://gist.github.com/benlansdell/44000c264d1b373c77497c0ea73f0ef2
# slightly modified
"""FilePicker for streamlit. 

Still doesn't seem to be a good solution for a way to select files to process from the server Streamlit is running on.

Here's a pretty functional solution. 

Usage:

```
import streamlit as st
from filepicker import st_file_selector

tif_file = st_file_selector(st, key = 'tif', label = 'Choose tif file')
```
"""

import os

import streamlit as st


def update_dir(key):
    global i_will_regret_this, i_will_regret_this2
    choice = st.session_state[key]
    if os.path.isdir(os.path.join(st.session_state[key + "curr_dir"], choice)):
        st.session_state[key + "index"] = 0
        st.session_state[key + "curr_dir"] = os.path.normpath(
            os.path.join(st.session_state[key + "curr_dir"], choice)
        )
        files = sorted(os.listdir(st.session_state[key + "curr_dir"]))
        files.insert(0, "..")
        files.insert(0, ".")
        st.session_state[key + "files"] = files


def st_file_selector(
    st_placeholder, path=".", label="Select a file/folder", key="selected"
):
    if key + "curr_dir" not in st.session_state:
        base_path = "." if path is None or path == "" else path
        base_path = (
            base_path if os.path.isdir(base_path) else os.path.dirname(base_path)
        )
        base_path = "." if base_path is None or base_path == "" else base_path

        files = sorted(os.listdir(base_path))
        files.insert(0, "..")
        files.insert(0, ".")
        st.session_state[key + "files"] = files
        st.session_state[key + "curr_dir"] = base_path
        st.session_state[key + "index"] = (
            st.session_state[key + "files"].index(os.path.basename(path))
            if os.path.isfile(path) and path[-4:] == '.pth'
            else 0
        )
    else:
        base_path = st.session_state[key + "curr_dir"]

    selected_file = st_placeholder.selectbox(
        label=label,
        options=st.session_state[key + "files"],
        index=st.session_state[key + "index"],
        key=key,
        on_change=lambda: update_dir(key),
    )
    selected_path = os.path.normpath(os.path.join(base_path, selected_file))
    st_placeholder.write(os.path.abspath(selected_path))

    return selected_path
