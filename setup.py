from setuptools import find_packages,setup

def get_requirements(file_name):
    """
    Return list of requirements
    """
    requirement = []
    with open(file_name) as obj:
        requirement=obj.readlines()
        requirement = [requires.replace("\n"," ") for requires in requirement]
        hypen_value = "-e ."
        if hypen_value in requirement:
            requirement.remove(hypen_value)
    return requirement


setup(

    name="CarPricePredictor",
    version="0.0.1",
    author="Saurabh",
    author_email="saurabhnair2391999@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements("requirements.txt")

)

