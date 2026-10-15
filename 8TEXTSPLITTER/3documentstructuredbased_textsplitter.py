#here differnent set of separators can be used for splitting to capture the meaningful chunking



from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = '''
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print(self.name, self.age)

class Student(Person):
    def __init__(self, name, age, roll_no):
        super().__init__(name, age)
        self.roll_no = roll_no

    def show(self):
        print(self.name, self.age, self.roll_no)

s1 = Student("Utkarsh", 20, 101)
s1.display()
s1.show()
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk)
    print("-" * 50)
