def main(args):
    for k,v in args.__dict__.items():
        print(k,v)

if __name__ == "__main__":
    # Default values (can be overridden by argparse)
    from auto_launcher import get_params
    args = get_params()
    main(args)