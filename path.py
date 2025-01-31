import os


class AppPath:
    current_dir = os.path.dirname(os.path.abspath(__file__))

    db_dir = os.path.join(current_dir, "db")

    data_dir = os.path.join(current_dir, "data")

    persistent_directory = os.path.join(db_dir, "my_db")
