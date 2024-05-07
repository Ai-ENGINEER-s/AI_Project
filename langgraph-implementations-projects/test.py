"---------Chat models import---------"
from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere 
"--------Embeddings models import -----------"
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)

"------Autre modules import --------------"
from typing_extensions import TypedDict


class Person(TypedDict):
    name:str 
    age : int 
    email:str 


def print_person_info(person:Person)->None: 
    print(f"Name: {person['name']}, Age: {person['age']}, Email: {person['email']}")

person_info: Person = {"name": "John", "age": 30, "email": "john@example.com"}
# print(person_info)


# Exercice 

# Exercice N°1

class Product(TypedDict):
    name:str 
    price:float 
    quantity:int 

def calculate_total_price(total_price:Product)->float:
    return total_price["price"] * total_price["quantity"]


product1:Product = {"name":"user1","price":8525.89, "quantity":89}


print(calculate_total_price(product1))


# utilisation de TypedDict : 

# Exercice N°2


class Adresses(TypedDict):
    city:str 
    state:str 
    zip_code :str 

def adress_info(adress:Adresses):
    return print(f"city : {adress['city']} state : {adress['state']}  zip_code : {adress['zip_code']}" )


adress1 : Adresses = {"city":"Ouagadougou", "state":"Burkina faso","zip_code":"Ouagadougou 2000 , Rue Sanmatenga"}

print(adress1)


# exercice 3 

class Book(TypedDict):
    title:str 
    author:str 
    isbn :str 


book1  : Book = {"title":"Les chaussures du roi MAKOKO","author":"BARRY","isbn":"ISBN20256"}

book2  : Book = {"title":"pere riche pere pauvre ","author":"BARRY","isbn":"ISBN20278"}

livre_store :list[Book] = [book1,book2]

for info in livre_store:
    book_title = info["title"]
    book_author= info["author"]
    book_isbn = info["isbn"]
    print(f"Author : {book_author}  title : {book_title} isBn : {book_isbn}")



# Exercice N°4 
# Exercice il faut les sir  either 


#  Exercice N°1 


from typing_extensions import TypedDict

class Student(TypedDict):
    age:int
    name:str
    major:int


def get_major_students(students:list[Student])->list[Student]:
    major_students = []
    for student in students :
        if student["major"] >= 21:
            major_students.append(student)

    return  major_students

student1: Student = {"age": 20, "name": "Sanoussa", "major": 21}
student2: Student = {"age": 25, "name": "John", "major": 20}
student3: Student = {"age": 22, "name": "Alice", "major": 22}

students = [student1, student2, student3]

print(get_major_students(students))


class User(TypedDict):
    username: str
    password: str
    email: str

def validate_user(user: User) -> bool:
    if len(user["password"]) < 8:
        print("Password must be at least 8 characters long.")
        return False
    elif "@" not in user["email"]:
        print("Invalid email format.")
        return False
    else:
        print("User information is valid.")
        return True
 
user1: User = {"username": "john_doe", "password": "pass1234", "email": "john@example.com"}
user2: User = {"username": "jane_doe", "password": "password", "email": "janeexample"}

validate_user(user1)
validate_user(user2)
