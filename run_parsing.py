# run_parsing.py
# This script properly initializes the absl.flags system.

from absl import app
from absl import flags

from slippi_db import parse_local

# The flags are already defined in parse_local when it's imported.
# We just need to get their values after parsing.
FLAGS = flags.FLAGS

def main(_):
  """This main function is called by app.run() after flags are parsed."""
  print(f"Starting parsing with the following flags:")
  print(f"  Replays Path: {FLAGS.replays_path}")
  print(f"  Output Path: {FLAGS.output_path}")
  print(f"  Recurse: {FLAGS.recurse}")
  print(f"  Workers: {FLAGS.workers}")

  parse_local.launch(
      workers=FLAGS.workers,
      replays_path=FLAGS.replays_path,
      output_path=FLAGS.output_path,
      recurse=FLAGS.recurse,
  )

if __name__ == '__main__':
  # This is the standard entrypoint for an absl app.
  app.run(main)