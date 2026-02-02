import config
from AgentFactory import AgentFactory


def main():
    factory = AgentFactory("meta-llama/Llama-3.1-8B-Instruct")
    agents, host = factory.build_prompt(verifier_decision="gen")
    msg = (
        "To find out how many DVDs Billy sold on Tuesday, we need to calculate the total number of DVDs "
        "sold to the first 3 customers and the next 2 customers.\n\n"
        "1. The first 3 customers buy one DVD each, so the total number of DVDs sold to them is 3 * 1 = 3 DVDs.\n"
        "2. The next 2 customers buy 2 DVDs each, so the total number of DVDs sold to them is 2 * 2 = 4 DVDs.\n"
        "3. The last 3 customers don't buy any DVDs, so the total number of DVDs sold to them is 0.\n\n"
        "Now, we add the total number of DVDs sold to the first 3 customers and the next 2 customers: 3 + 4 = 7 DVDs.\n\n"
        "Therefore, Billy sold 7 DVDs on Tuesday.\n\n"
        "ANSWER: 7"
    )
    task = host.build_chat_prompt(config.ROLE_SYSTEM_PROMPTS["verifier"], msg)
    print(task)
    print("-------")
    print(agents["verifier"].generate(task))


if __name__ == "__main__":
    main()
