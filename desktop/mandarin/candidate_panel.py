"""A tiny Cocoa view adapter: messages replace cards without parsing edited text."""
class CandidatePanel:
    def __init__(self, message, message_scroll, cards, editors):
        self.message = message
        self.message_scroll = message_scroll
        self.cards = cards
        self.editors = editors

    def setString_(self, text):
        self.message.setString_(text)
        self.message_scroll.setHidden_(not bool(text))
        for card in self.cards:
            card.setHidden_(bool(text))
        for editor in self.editors:
            editor.setString_('')
            editor.setEditable_(False)
            manager = editor.undoManager()
            if manager is not None:
                manager.removeAllActions()

    def string(self):
        return self.message.string()

    def show_choices(self):
        self.message.setString_('')
        self.message_scroll.setHidden_(True)
        for card in self.cards:
            card.setHidden_(False)
