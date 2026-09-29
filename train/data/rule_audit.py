"""Independent, input-only label checks for the Open-Jev browser-control and reasoning-control families.

These implement the visible task rules, not the source generator or its hidden
metadata. Source labels are compared only after deriving the expected answer.
No source-provided Python program is executed: the small integer language is
interpreted with an AST whitelist.
"""
import ast
import datetime
import operator
import re


BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
COMPARE = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}


def expression(node, env):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, bool)):
            raise ValueError('Only integer/Boolean constants are allowed')
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [expression(n, env) for n in node.elts]
    if isinstance(node, ast.UnaryOp):
        value = expression(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.Not):
            return not value
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        return BINARY[type(node.op)](expression(node.left, env), expression(node.right, env))
    if isinstance(node, ast.Compare):
        values = [expression(node.left, env)] + [expression(n, env) for n in node.comparators]
        return all(COMPARE[type(op)](left, right)
                   for op, left, right in zip(node.ops, values, values[1:]))
    if isinstance(node, ast.Subscript):
        return expression(node.value, env)[expression(node.slice, env)]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'range':
        if node.keywords:
            raise ValueError('range keywords are unsupported')
        values = list(range(*[expression(n, env) for n in node.args]))
        if len(values) > 1000:
            raise ValueError('Loop bound exceeded')
        return values
    raise ValueError('Unsupported expression: ' + ast.dump(node))


def statements(nodes, env):
    for node in nodes:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError('Unsupported assignment')
            env[node.targets[0].id] = expression(node.value, env)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            env[node.target.id] = BINARY[type(node.op)](env[node.target.id], expression(node.value, env))
        elif isinstance(node, ast.If):
            statements(node.body if expression(node.test, env) else node.orelse, env)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            values = expression(node.iter, env)
            if len(values) > 1000 or node.orelse:
                raise ValueError('Unsupported loop')
            for value in values:
                env[node.target.id] = value
                statements(node.body, env)
        else:
            raise ValueError('Unsupported statement: ' + ast.dump(node))


def boolean(text, env):
    tokens = re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[()]', text)
    if ''.join(tokens) != re.sub(r'\s+', '', text):
        raise ValueError('Unsupported Boolean syntax')
    pos = 0
    precedence = {'IMPLIES': 1, 'OR': 2, 'XOR': 3, 'AND': 4}

    def parse(minimum=0):
        nonlocal pos
        token = tokens[pos]
        pos += 1
        if token == 'NOT':
            left = not parse(5)
        elif token == '(':
            left = parse()
            if tokens[pos] != ')':
                raise ValueError('Unbalanced Boolean expression')
            pos += 1
        else:
            left = env[token]
        while pos < len(tokens) and tokens[pos] in precedence and precedence[tokens[pos]] >= minimum:
            op = tokens[pos]
            pos += 1
            right = parse(precedence[op] + (0 if op == 'IMPLIES' else 1))
            if op == 'AND':
                left = left and right
            elif op == 'OR':
                left = left or right
            elif op == 'XOR':
                left = left != right
            else:
                left = (not left) or right
        return left

    answer = parse()
    if pos != len(tokens):
        raise ValueError('Trailing Boolean input')
    return answer


def reasoning_gold(row):
    state, question, kind = row['state'], row['question'], row['kind']
    # Select the visible task by its input fields, not privileged metadata.
    if 'expression' in state and 'values' in state:
        value = expression(ast.parse(state['expression'], mode='eval').body, state['values'])
        answer, description = str(value), f'The exact integer result is {value}'
        score = (value > 0) - (value < 0) + 1
    elif 'source' in state:
        env = {}
        statements(ast.parse(state['source']).body, env)
        value = env['result']
        answer, description = str(value), f'The exact integer result is {value}'
        score = (value > 0) - (value < 0) + 1
    elif 'events' in state and 'cutoff' in state:
        events = [(datetime.datetime.fromisoformat(e['timestamp']), e['id']) for e in state['events']]
        choose = max if 'latest' in question else min
        answer = choose(events)[1]
        description = f'Event {answer}'
        score = sum(t < datetime.datetime.fromisoformat(state['cutoff']) for t, _ in events)
    elif 'divisor' in state:
        values = sorted([v for v in state['values'] if v % state['divisor'] == 0],
                        reverse=state['order'] == 'descending')
        index = state['zero_based_index']
        value = values[index] if 0 <= index < len(values) else None
        answer = str(value) if value is not None else 'not_present'
        description = f'The exact integer result is {value}' if value is not None else 'The requested result does not exist'
        score = min(len(values), 3)
    elif 'directed_edges' in state:
        edges, start, goal = state['directed_edges'], state['start'], state['goal']
        seen, pending = {start}, [start]
        while pending:
            current = pending.pop()
            for left, right in edges:
                if left == current and right not in seen:
                    seen.add(right)
                    pending.append(right)
        answer = 'direct' if [start, goal] in edges else 'indirect' if goal in seen else 'unreachable'
        description = {'direct': 'A direct start-to-goal edge exists',
                       'indirect': 'A path exists but no direct edge exists',
                       'unreachable': 'No directed path exists'}[answer]
        score = len(seen - {start})
    elif 'query_entity' in state:
        def normalize(entity):
            seen = set()
            while entity in state['aliases']:
                if entity in seen:
                    raise ValueError('Alias cycle')
                seen.add(entity)
                entity = state['aliases'][entity]
            return entity
        eligible = [r for r in state['records']
                    if normalize(r['entity']) == normalize(state['query_entity'])
                    and r['property'] == state['query_property']
                    and r['authoritative'] and not r['retracted']]
        revision = max((r['revision'] for r in eligible), default=None)
        latest = [r for r in eligible if r['revision'] == revision]
        values = {r['value'] for r in latest}
        answer = 'not_stated' if not values else 'conflict' if len(values) > 1 else next(iter(values))
        description = ('No eligible status record remains' if answer == 'not_stated' else
                       'Highest eligible revision contains conflicting values' if answer == 'conflict' else
                       f'Current status is {answer}')
        score = min(len(latest), 3)
    elif 'assignments' in state:
        answer = 'true' if boolean(state['expression'], state['assignments']) else 'false'
        description = f'The expression evaluates to {answer}'
        score = sum(state['assignments'].values())
    else:
        raise ValueError('Unknown reasoning input')
    if kind == 'score':
        return score
    if kind == 'choice':
        return [o.split(': ', 1)[0] for o in row['options']].index(answer)
    proposed = question.split('Proposed answer: ', 1)[1].split('\n', 1)[0]
    return int(proposed == description)


def browser_gold(row):
    import json
    state, question = row['state'], row['question']
    goal = state['goal'].split('\n')[0]
    values = re.findall(r'"([^"]*)"', goal)
    if goal.startswith('Turn on '):
        operation, requested = 'CLICK', True
        label, scope = values
    elif goal.startswith('Select '):
        operation = 'SELECT'
        requested, label, scope = values
    elif goal.startswith('Enter '):
        operation = 'TYPE_TEXT'
        requested, label, scope = values
    else:
        raise ValueError('Unsupported browser goal')
    matches = [e for e in state['snapshot']['elements']
               if e['label'] == label and e['scope_path'] == [scope]]
    element = matches[0] if len(matches) == 1 else None
    names = [o.split(': ', 1)[0] for o in row['options']]
    if question.startswith('Choose the next browser operation'):
        done = element is not None and (
            element.get('aria_checked') is True if operation == 'CLICK' else
            element.get('value') == requested if operation == 'TYPE_TEXT' else
            element.get('options', {}).get(element.get('selected_option_id')) == requested)
        busy = any(s.get('aria_busy') is True and s['scope_path'] == [scope]
                   for s in state['snapshot']['sections'])
        value_ids = [] if element is None else [key for key, value in
            (element.get('options', {}) if operation == 'SELECT' else
             state.get('text_candidates', {}).get(element['id'], {})).items() if value == requested]
        if done:
            answer = 'DONE'
        elif busy:
            answer = 'WAIT'
        elif (element is None or operation not in element['actions'] or operation not in names
              or (operation != 'CLICK' and len(value_ids) != 1)):
            answer = 'BLOCKED'
        else:
            answer = operation
    elif question.startswith('If the operation is '):
        if element is None:
            raise ValueError('Target is not unique')
        answer = element['id']
    else:
        spec = json.loads(question)
        target = next(e for e in state['snapshot']['elements'] if e['id'] == spec['element_id'])
        candidates = target['options'] if 'dropdown' in spec['task'] else state['text_candidates'][target['id']]
        matching = [key for key, value in candidates.items() if value == requested]
        if len(matching) != 1:
            raise ValueError('Value is not unique')
        answer = matching[0]
    return names.index(answer)
