""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import symbols, ctc_symbols


# Mappings from symbol to numeric ID and vice versa:
symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
_ctc_symbole_to_id = {s: i for i, s in enumerate(ctc_symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  return sequence

def sequence_to_ctc_sequence(sequence):
  return [_ctc_symbole_to_id[_id_to_symbol[s]] for s in sequence if _id_to_symbol[s] in ctc_symbols]

def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    # Пропускаем пустые имена
    if not name or not name.strip():
      continue
    
    name = name.strip()  # Убираем лишние пробелы
    
    try:
      cleaner = getattr(cleaners, name)
      if not cleaner:
        raise Exception('Unknown cleaner: %s' % name)
      text = cleaner(text)
    except AttributeError as e:
      print(f"Error: cleaner '{name}' not found in cleaners module")
      print(f"Available cleaners: {[attr for attr in dir(cleaners) if not attr.startswith('_')]}")
      raise Exception('Unknown cleaner: %s' % name) from e
  return text


def _symbols_to_sequence(symbols):
  return [symbol_to_id[s] for s in symbols]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in symbol_to_id and s != '_' and s != '~'


def symbol_is_valid(s):
    # Допустимы все символы, кроме '_' и '~'
    return s in symbol_to_id and s != '_' and s != '~'


def get_arpabet(word, dictionary):
  word_arp = dictionary.lookup(word)
  if word_arp is not None:
    return '{%s}' % (word_arp)
  else:
    return None
